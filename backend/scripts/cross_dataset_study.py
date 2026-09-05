"""Does a learned forecaster beat persistence? Tested across five workloads.

`horizon_study.py` answers this descriptively on one dataset: it reports a mean
margin over three windows. That is not enough to make a claim. Three windows
cannot distinguish a real effect from noise, and a result established on one
trace says nothing about cloud workloads in general - as this project found the
hard way, when a conclusion drawn from synthetic data reversed on real data.

This script makes the claim testable:

**Rolling-origin walk-forward evaluation.** Instead of one 70/15/15 split, the
model is refit at K successive origins, each time on all history before the test
block - which is what a deployed forecaster does. Test blocks are *disjoint*, so
the K paired margins are close to independent observations rather than K views of
the same test set.

Training-set size therefore grows across blocks. That is deliberate. Restricting
it to a short fixed window handicaps only the learned arms, because persistence
needs no training at all: on synthetic demand XGBoost loses to persistence with a
7-day window (MAE ratio 1.25) and beats it with three weeks. `--sliding`
reproduces the fixed-window variant as a robustness check.

**Paired significance test.** For each (dataset, horizon, model) the K margins go
through a two-sided Wilcoxon signed-rank test against zero. Signed-rank rather
than a t-test because K is small and per-block R² is not normally distributed.

**Multiple-comparison correction.** 5 datasets x 4 horizons x 3 models = 60
hypotheses. At alpha=0.05 three would clear by chance alone, so raw p-values are
adjusted by Holm-Bonferroni and both are reported.

**Two error measures.** R² is reported because the rest of the project reports
it, but R² has a known pathology on near-random-walk series: it is computed
against the variance of the test block, which changes with the block. The MAE
ratio (model MAE / persistence MAE, <1 means the model wins) is scale-free and
block-independent, and is the measure the conclusion rests on.

Only CPU demand is modelled. The Azure trace has no memory column - its RAM is
derived - so a RAM claim spanning all five datasets would not be honest.

    python scripts/cross_dataset_study.py
    python scripts/cross_dataset_study.py --datasets bitbrains google --horizons 1 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.ml.predictor as predictor_module  # noqa: E402
from app.ml.predictor import FEATURES, SplitMetrics, WorkloadPredictor, prepare_features  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"

HORIZONS = [1, 3, 6, 12]
ALGOS = ["xgboost", "rf", "lr"]

DATASETS = {
    "synthetic": (DATA / "workload_history.csv", "Generator written for this project"),
    "bitbrains": (DATA / "workload_bitbrains.csv", "Bitbrains GWA-T-12, enterprise VMs, 2013"),
    "google": (DATA / "workload_google.csv", "Google Borg cluster tasks, 2019"),
    "azure": (DATA / "workload_azure.csv", "Microsoft Azure VM CPU readings"),
    "alibaba": (DATA / "workload_alibaba.csv", "Alibaba production cluster machines, 2018"),
}


# ---------------------------------------------------------------------------
# Trace characterisation
# ---------------------------------------------------------------------------

def describe(cpu: np.ndarray) -> dict:
    """Cheap statistics that might predict whether a model can beat persistence.

    `diff_acf1` - the lag-1 autocorrelation of the *first difference* - is the
    one that matters. Near zero means a random walk: this interval's change tells
    you nothing about the next one, so persistence is the optimal forecast and no
    model can do better. Strongly negative means changes reverse, which
    persistence cannot exploit and a model can.
    """
    d = np.diff(cpu)
    return {
        "n": int(len(cpu)),
        "mean": round(float(cpu.mean()), 3),
        "cv": round(float(cpu.std() / cpu.mean()), 4),
        "acf1": round(float(np.corrcoef(cpu[:-1], cpu[1:])[0, 1]), 4),
        "acf12": round(float(np.corrcoef(cpu[:-12], cpu[12:])[0, 1]), 4),
        "diff_acf1": round(float(np.corrcoef(d[:-1], d[1:])[0, 1]), 4),
        "mean_abs_step_pct": round(float(np.abs(d).mean() / cpu.mean() * 100), 3),
    }


# ---------------------------------------------------------------------------
# Rolling-origin evaluation
# ---------------------------------------------------------------------------

# Seven days of 5-minute samples: the warm-up before the first test block, and
# the window length under --sliding. A full week matters because `day_of_week` is
# a model feature, so anything shorter leaves that feature untrained. It also
# leaves ~23 disjoint test blocks in a 30-day trace, which is enough for a
# signed-rank test to have power.
DEFAULT_TRAIN_WINDOW = 2016


def plan_blocks(n: int, train_window: int = DEFAULT_TRAIN_WINDOW) -> tuple[int, int, int]:
    """Choose (train_len, test_len, n_blocks) for a trace of n usable rows.

    Blocks must be numerous enough for a signed-rank test to have any power, and
    long enough that a per-block R² is not dominated by a handful of points.
    """
    test_len = max(72, min(288, n // 30))            # 6 h .. 24 h
    train_len = max(288, min(train_window, n // 3))  # never more than a third
    n_blocks = min(24, (n - train_len) // test_len)
    return train_len, test_len, int(n_blocks)


def evaluate(frame: pd.DataFrame, algo: str,
             train_window: int = DEFAULT_TRAIN_WINDOW,
             expanding: bool = True, n_jobs: int = 1) -> list[dict]:
    """Refit at successive origins; return one record per disjoint test block.

    `expanding` (the default) trains block k on *everything* before it, which is
    what a deployed system does - retrain on all history, forecast the next
    period. `expanding=False` slides a fixed-length window instead, holding
    training-set size constant across blocks.

    The expanding window is primary because a fixed short window silently
    handicaps the models that need data: on synthetic demand, XGBoost loses to
    persistence at one step with a 7-day window (MAE ratio 1.25) but beats it
    with three weeks. A conclusion drawn from the short window would be measuring
    the training budget, not the workload. Persistence needs no training at all,
    so any restriction on training data is a restriction on one arm only.
    """
    feat = prepare_features(frame)
    n = len(feat)
    train_len, test_len, n_blocks = plan_blocks(n, train_window)
    if n_blocks < 6:
        return []

    X_all = feat[FEATURES]
    y_all = feat["target_cpu"]
    persist_all = feat["cpu_demand"]
    out = []

    for k in range(n_blocks):
        origin = train_len + k * test_len
        tr = slice(0 if expanding else origin - train_len, origin)
        te = slice(origin, origin + test_len)

        # `n_jobs` is a speed knob only. The deployed predictor pins it to 1
        # because the target container has 512 MB and extra worker threads cost
        # more memory than they save; this study is a local research run over
        # ~1,400 fits, where that trade-off is reversed. Both forests are
        # seeded, so the fitted models are the same either way.
        model = WorkloadPredictor(algo)._make_model(
            {} if algo == "lr" else {"n_jobs": n_jobs})
        model.fit(X_all.iloc[tr], y_all.iloc[tr])
        y_te = y_all.iloc[te].to_numpy()

        m = SplitMetrics.compute(y_te, model.predict(X_all.iloc[te]))
        p = SplitMetrics.compute(y_te, persist_all.iloc[te].to_numpy())
        out.append({
            "block": k,
            "n_train": tr.stop - tr.start,
            "model_r2": m.r2, "persistence_r2": p.r2,
            "model_mae": m.mae, "persistence_mae": p.mae,
            "r2_margin": m.r2 - p.r2,
            "mae_ratio": m.mae / p.mae if p.mae > 1e-12 else float("nan"),
        })

        del model
    return out


def wilcoxon(diffs: list[float]) -> tuple[float, float]:
    """Two-sided signed-rank test against zero. Returns (statistic, p)."""
    from scipy.stats import wilcoxon as _w

    nz = [d for d in diffs if abs(d) > 1e-12]
    if len(nz) < 6:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = _w(nz, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, monotone-enforced."""
    idx = [i for i, p in enumerate(pvalues) if p == p]     # drop NaN
    m = len(idx)
    adj = [float("nan")] * len(pvalues)
    order = sorted(idx, key=lambda i: pvalues[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


ALPHA = 0.05


def classify(row: dict) -> str:
    """The verdict for one (dataset, horizon, model) cell.

    Direction comes from the *median* ratio, not the mean, because the
    signed-rank test is a statement about the median. Reading direction off the
    mean can contradict the very test that established significance: a few badly
    mispredicted blocks drag the mean above 1.0 while the model still wins the
    majority of blocks. Google at one step is exactly that shape - mean ratio
    1.017, yet 17 of 23 blocks won.

    Lives here rather than inline so `render_cross_dataset.py` can apply the same
    rule to a stored result instead of restating it.
    """
    p = row.get("p_mae_holm")
    if p is None or p != p or p >= ALPHA:
        return "no difference"
    return "model" if row["mae_ratio_median"] < 1.0 else "persistence"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--train-window", type=int, default=DEFAULT_TRAIN_WINDOW,
                    help="warm-up length before the first test block, and the "
                         "window length when --sliding is given (2016 = 7 days)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="threads per model fit (-1 = all cores). Speed only; "
                         "the deployed predictor pins this to 1 for memory.")
    ap.add_argument("--sliding", action="store_true",
                    help="hold training-set size fixed instead of expanding it; "
                         "a robustness check, not the primary design")
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "cross_dataset_study.json")
    args = ap.parse_args()

    started = time.time()
    original_horizon = predictor_module.HORIZON

    frames, traits = {}, {}
    for name in args.datasets:
        path, desc = DATASETS[name]
        if not path.exists():
            print(f"  skipping {name}: {path} not found", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        frames[name] = df
        traits[name] = {"description": desc, "source": path.name,
                        **describe(df["cpu_demand"].to_numpy(float))}

    if not frames:
        print("ERROR: no datasets available.", file=sys.stderr)
        return 1

    print("Trace characteristics\n" + "-" * 78)
    print(f"{'trace':<11}{'rows':>7}{'mean cpu':>10}{'CV':>8}{'acf1':>8}"
          f"{'diff_acf1':>11}{'step %':>9}")
    for name, t in traits.items():
        print(f"{name:<11}{t['n']:>7}{t['mean']:>10.1f}{t['cv']:>8.3f}"
              f"{t['acf1']:>8.3f}{t['diff_acf1']:>11.3f}{t['mean_abs_step_pct']:>9.2f}")

    total = len(frames) * len(args.horizons) * len(args.algos)
    print(f"\nRolling-origin study: {total} cells "
          f"({len(frames)} traces x {len(args.horizons)} horizons x {len(args.algos)} models)")
    print(f"Disjoint test blocks, "
          f"{'fixed-length sliding' if args.sliding else 'expanding'} training window, "
          f"CPU demand only.\n")

    rows: list[dict] = []
    done = 0
    for name, df in frames.items():
        n_after = len(prepare_features(df))
        train_len, test_len, n_blocks = plan_blocks(n_after, args.train_window)
        print(f"{name}: {n_blocks} blocks of {test_len} rows "
              f"({test_len * args.interval / 60:.0f} h), training window {train_len} rows "
              f"({train_len * args.interval / 1440:.1f} d)")

        for h in args.horizons:
            predictor_module.HORIZON = h
            for algo in args.algos:
                blocks = evaluate(df, algo, args.train_window,
                                  not args.sliding, args.n_jobs)
                done += 1
                if not blocks:
                    print(f"  h={h:>2} {algo:<8} too few blocks; skipped")
                    continue

                r2m = [b["r2_margin"] for b in blocks]
                ratio = [b["mae_ratio"] for b in blocks]
                _, p_r2 = wilcoxon(r2m)
                _, p_mae = wilcoxon([r - 1.0 for r in ratio])

                row = {
                    "dataset": name, "horizon_intervals": h,
                    "horizon_minutes": h * args.interval, "algo": algo,
                    "blocks": len(blocks),
                    "n_train_first": blocks[0]["n_train"],
                    "n_train_last": blocks[-1]["n_train"],
                    "model_r2_mean": round(float(np.mean([b["model_r2"] for b in blocks])), 4),
                    "persistence_r2_mean": round(
                        float(np.mean([b["persistence_r2"] for b in blocks])), 4),
                    "r2_margin_mean": round(float(np.mean(r2m)), 4),
                    "r2_margin_median": round(float(np.median(r2m)), 4),
                    "r2_wins": int(sum(1 for x in r2m if x > 0)),
                    "mae_ratio_mean": round(float(np.mean(ratio)), 4),
                    "mae_ratio_median": round(float(np.median(ratio)), 4),
                    "mae_wins": int(sum(1 for x in ratio if x < 1.0)),
                    "p_r2": p_r2, "p_mae": p_mae,
                }
                rows.append(row)
                print(f"  h={h:>2} ({row['horizon_minutes']:>2}m) {algo:<8} "
                      f"MAE ratio {row['mae_ratio_mean']:.3f} "
                      f"({row['mae_wins']}/{row['blocks']} blocks) "
                      f"p={p_mae:.2e}   R2 margin {row['r2_margin_mean']:+.4f}"
                      f"   [{done}/{total}]", flush=True)
        print()

    predictor_module.HORIZON = original_horizon

    # Holm-Bonferroni over the whole family of tests, not per dataset: the
    # 60 hypotheses were all generated by one study and are reported together.
    for key, adj_key in (("p_mae", "p_mae_holm"), ("p_r2", "p_r2_holm")):
        for row, adj in zip(rows, holm([r[key] for r in rows])):
            row[adj_key] = adj

    for row in rows:
        row["verdict"] = classify(row)

    # --------------------------------------------------------------- summary
    print("=" * 78)
    print("VERDICT BY TRACE  (best model per horizon, MAE ratio, Holm-adjusted)")
    print("=" * 78)
    print(f"{'trace':<11}{'diff_acf1':>10}  " +
          "".join(f"{h * args.interval:>4}m" for h in args.horizons))
    for name in frames:
        cells = []
        for h in args.horizons:
            cand = [r for r in rows if r["dataset"] == name and r["horizon_intervals"] == h]
            if not cand:
                cells.append("   -")
                continue
            best = min(cand, key=lambda r: r["mae_ratio_median"])
            mark = {"model": "+", "persistence": "-", "no difference": "="}[best["verdict"]]
            cells.append(f"{mark}{best['mae_ratio_mean']:>4.2f}")
        print(f"{name:<11}{traits[name]['diff_acf1']:>10.3f}  " +
              "".join(f"{c:>5}" for c in cells))
    print("\n  +  a learned model beats persistence, significant after Holm correction")
    print("  -  persistence beats every learned model, significant")
    print("  =  no significant difference")

    # Is the outcome predicted by the trace's first-difference autocorrelation?
    per_trace = []
    for name in frames:
        cand = [r for r in rows if r["dataset"] == name]
        if cand:
            per_trace.append((traits[name]["diff_acf1"],
                              float(np.mean([r["mae_ratio_mean"] for r in cand]))))
    correlation = (round(float(np.corrcoef([a for a, _ in per_trace],
                                           [b for _, b in per_trace])[0, 1]), 4)
                   if len(per_trace) > 2 else None)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {
            "design": ("rolling-origin walk-forward, disjoint test blocks, "
                       + ("fixed-length sliding" if args.sliding else "expanding")
                       + " training window"),
            "target": "cpu_demand only (Azure has no measured memory column)",
            "tuning": "none - library defaults for every arm, so the comparison is "
                      "like-for-like and not an artefact of unequal search budgets",
            "test": "two-sided Wilcoxon signed-rank on per-block paired differences",
            "correction": "Holm-Bonferroni across all reported tests",
            "primary_measure": "MAE ratio (model / persistence); <1 means the model wins",
            "interval_minutes": args.interval,
            "n_jobs": args.n_jobs,
            "train_window_intervals": args.train_window,
            "expanding_window": not args.sliding,
            "train_window_note": ("the expanding window is primary because it is what a "
                                  "deployed system does and because a short fixed window "
                                  "restricts only the learned arms - persistence needs no "
                                  "training. --sliding reproduces the fixed-window check."),
            "horizons": args.horizons,
            "algos": args.algos,
        },
        "traces": traits,
        "rows": rows,
        "diff_acf1_vs_mae_ratio_correlation": correlation,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if correlation is not None:
        print(f"\nCorrelation between a trace's diff_acf1 and its mean MAE ratio: "
              f"{correlation:+.3f}  (n={len(per_trace)} traces)")
    print(f"\nWrote {args.out} in {payload['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
