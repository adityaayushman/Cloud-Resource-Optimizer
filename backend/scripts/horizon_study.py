"""Does the learned forecaster actually beat a naive baseline?

A high R² on an autocorrelated series proves nothing on its own: "next interval
equals this one" already scores ~0.93 on this workload. The honest question is
whether the model beats *that*, and the answer depends entirely on how far ahead
it has to see.

This script sweeps the forecast horizon and reports the margin over persistence
for each model, averaged across seeds. Output: `artifacts/horizon_study.json`.

    python scripts/horizon_study.py --seeds 7 42 99 --days 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.ml.predictor as predictor_module  # noqa: E402
from app.ml.predictor import WorkloadPredictor  # noqa: E402
from app.workload import build_dataset  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
HORIZONS = [1, 3, 6, 12]
ALGOS = ["xgboost", "rf", "lr"]


def summarise(results: list[dict], breakeven: int | None) -> str:
    """Describe what the numbers actually show.

    An earlier version wrote a fixed sentence asserting that the forecaster's
    advantage grows with horizon. That was true of the synthetic workload it was
    written against and false on Bitbrains, where the deficit grows instead - so
    the file cheerfully contradicted the table printed directly above it. The
    summary is derived from the rows now.
    """
    xgb = sorted((r for r in results if r["algo"] == "xgboost"),
                 key=lambda r: r["horizon_intervals"])
    if not xgb:
        return "No results."

    first, last = xgb[0]["margin_mean"], xgb[-1]["margin_mean"]
    trend = ("grows with horizon" if last > first + 0.01
             else "shrinks with horizon" if last < first - 0.01
             else "is flat across horizons")

    if breakeven is not None:
        stance = (f"XGBoost beats persistence on every seed from {breakeven} minutes "
                  f"ahead onward")
    elif all(r["margin_mean"] < 0 for r in xgb):
        stance = ("XGBoost does not beat persistence at any horizon tested, and is "
                  "behind it on average everywhere")
    else:
        stance = ("XGBoost beats persistence on some seeds but not on all of them at "
                  "any horizon tested")

    return (f"{stance}. Its margin over the baseline {trend} "
            f"({first:+.4f} at the shortest horizon, {last:+.4f} at the longest).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--trace", type=Path, default=None,
                    help="use a recorded trace instead of the synthetic generator; "
                         "seeds then select different windows of it")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    original_horizon = predictor_module.HORIZON
    started = time.time()
    results: list[dict] = []

    if args.trace:
        # A recorded trace has no randomness, so "seeds" select different
        # contiguous windows of it. Each window is a distinct month-scale
        # sample of the same production workload.
        full = pd.read_csv(args.trace)
        span = len(full)
        width = max(2000, span // 2)
        frames = {}
        for i, s in enumerate(args.seeds):
            start = int((i / max(1, len(args.seeds))) * (span - width))
            frames[s] = full.iloc[start:start + width].reset_index(drop=True)
        print(f"Trace: {args.trace.name}, {span:,} rows -> "
              f"{len(args.seeds)} windows of {width:,}")
    else:
        frames = {s: pd.DataFrame(build_dataset(days=args.days, seed=s,
                                                interval_minutes=args.interval))
                  for s in args.seeds}

    print(f"Horizon study: {len(HORIZONS)} horizons x {len(ALGOS)} models "
          f"x {len(args.seeds)} seeds ({args.days} days each)\n")

    for h in HORIZONS:
        predictor_module.HORIZON = h
        for algo in ALGOS:
            margins, model_r2, naive_r2, maes = [], [], [], []
            for seed in args.seeds:
                report = WorkloadPredictor(algo).train(frames[seed], tune=False)
                cpu = report["targets"]["cpu_demand_t+1"]
                m, n = cpu["test"]["r2"], cpu["naive_persistence_test"]["r2"]
                margins.append(m - n)
                model_r2.append(m)
                naive_r2.append(n)
                maes.append(cpu["test"]["mae"])

            row = {
                "horizon_intervals": h,
                "horizon_minutes": h * args.interval,
                "algo": algo,
                "model_r2_mean": round(statistics.mean(model_r2), 4),
                "persistence_r2_mean": round(statistics.mean(naive_r2), 4),
                "margin_mean": round(statistics.mean(margins), 4),
                "margin_sd": round(statistics.stdev(margins), 4) if len(margins) > 1 else 0.0,
                "mae_mean": round(statistics.mean(maes), 4),
                "wins": sum(1 for x in margins if x > 0),
                "seeds": len(args.seeds),
            }
            results.append(row)
            print(f"  h={h:>2} ({row['horizon_minutes']:>2} min)  {algo:<8} "
                  f"R2={row['model_r2_mean']:.4f}  persistence={row['persistence_r2_mean']:.4f}  "
                  f"margin={row['margin_mean']:+.4f}  wins {row['wins']}/{row['seeds']}")
        print()

    predictor_module.HORIZON = original_horizon

    xgb = [r for r in results if r["algo"] == "xgboost"]
    breakeven = next((r["horizon_minutes"] for r in xgb if r["wins"] == r["seeds"]), None)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {
            "seeds": args.seeds,
            "days": args.days,
            "interval_minutes": args.interval,
            "horizons": HORIZONS,
            "workload": "real_trace" if args.trace else "synthetic",
            "trace": str(args.trace.name) if args.trace else None,
            "note": (
                "Margin is model R2 minus persistence-baseline R2 on the held-out "
                "test block, averaged over seeds. Positive means the learned model "
                "beats 'next interval equals this one'."
            ),
        },
        "rows": results,
        "conclusion": {
            "breakeven_horizon_minutes": breakeven,
            "summary": summarise(results, breakeven),
            "caveat": (
                "Descriptive only. Seeds here are overlapping windows of one "
                "workload and no significance test is applied, so a small margin "
                "is not evidence. scripts/cross_dataset_study.py answers the same "
                "question across five workloads with disjoint test blocks, a "
                "Wilcoxon signed-rank test and Holm-Bonferroni correction."
            ),
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    out = args.out or (ARTIFACTS / "horizon_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} in {payload['elapsed_seconds']}s")
    # Printing "first beats persistence at None minutes" was the visible symptom
    # of the summary being asserted rather than derived.
    print(payload["conclusion"]["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
