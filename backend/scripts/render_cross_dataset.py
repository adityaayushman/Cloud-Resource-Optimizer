"""Render `cross_dataset_study.json` into the Markdown tables the docs use.

Sixty cells is more than anyone should transcribe by hand into a report, and a
mis-typed p-value in a results table is indistinguishable from a lie. This emits
the tables directly from the study output so `docs/RESULTS-CROSS-DATASET.md` can
be regenerated rather than maintained.

    python scripts/render_cross_dataset.py >> ../docs/RESULTS-CROSS-DATASET.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _study_module():
    """Import the study script so the verdict rule has exactly one definition.

    Restating it here would let the tables drift from the analysis that produced
    them - the failure mode this renderer exists to prevent.
    """
    spec = importlib.util.spec_from_file_location(
        "cross_dataset_study", Path(__file__).parent / "cross_dataset_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cross_dataset_study"] = module
    spec.loader.exec_module(module)
    return module


classify = _study_module().classify

MARK = {"model": "**beats**", "persistence": "loses", "no difference": "ties"}


def fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "—"
    if p < 1e-4:
        return f"{p:.0e}".replace("e-0", "e−").replace("e-", "e−")
    return f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path,
                    default=ROOT / "artifacts" / "cross_dataset_study.json")
    args = ap.parse_args()

    study = json.loads(args.json.read_text(encoding="utf-8"))
    rows = study["rows"]
    # Re-derive rather than trust the stored field, so a result produced before a
    # rule change still renders under the current rule.
    for row in rows:
        row["verdict"] = classify(row)
    traces = study["traces"]
    horizons = sorted({r["horizon_minutes"] for r in rows})
    algos = sorted({r["algo"] for r in rows}, key=["xgboost", "rf", "lr"].index)
    label = {"xgboost": "XGBoost", "rf": "Random Forest", "lr": "Linear Regression"}

    # ---- headline: best model per (trace, horizon) --------------------------
    print("### Verdict by workload\n")
    print("MAE ratio of the *best* model at each horizon (model ÷ persistence; "
          "below 1 means the model wins), with the Holm-adjusted verdict.\n")
    print("| trace | diff_acf1 | " + " | ".join(f"{h} min" for h in horizons) + " |")
    print("|---|---|" + "---|" * len(horizons))
    for name, t in traces.items():
        cells = []
        for h in horizons:
            cand = [r for r in rows
                    if r["dataset"] == name and r["horizon_minutes"] == h]
            if not cand:
                cells.append("—")
                continue
            best = min(cand, key=lambda r: r["mae_ratio_median"])
            tick = {"model": "✅", "persistence": "❌", "no difference": "➖"}[best["verdict"]]
            cells.append(f"{tick} {best['mae_ratio_median']:.2f}")
        print(f"| **{name}** | {t['diff_acf1']:+.3f} | " + " | ".join(cells) + " |")
    print("\n✅ a learned model beats persistence, significant after Holm correction · "
          "❌ persistence beats every model, significant · ➖ no significant difference\n")

    # ---- per-trace detail ---------------------------------------------------
    for name in traces:
        subset = [r for r in rows if r["dataset"] == name]
        if not subset:
            continue
        blocks = subset[0]["blocks"]
        print(f"\n#### {name} — {traces[name]['description']}")
        print(f"\n*{blocks} disjoint test blocks · diff_acf1 = "
              f"{traces[name]['diff_acf1']:+.3f}*\n")
        print("| Horizon | Model | MAE ratio | Blocks won | p (raw) | p (Holm) | Verdict |")
        print("|---|---|---|---|---|---|---|")
        for h in horizons:
            for algo in algos:
                r = next((x for x in subset if x["horizon_minutes"] == h
                          and x["algo"] == algo), None)
                if r is None:
                    continue
                print(f"| {h} min | {label[algo]} | {r['mae_ratio_mean']:.3f} | "
                      f"{r['mae_wins']}/{r['blocks']} | {fmt_p(r['p_mae'])} | "
                      f"{fmt_p(r['p_mae_holm'])} | {MARK[r['verdict']]} |")

    # ---- does the diagnostic predict the outcome? ---------------------------
    print("\n### Does `diff_acf1` predict the outcome?\n")
    print("| trace | diff_acf1 | mean MAE ratio | cells won | cells lost | cells tied |")
    print("|---|---|---|---|---|---|")
    for name, t in traces.items():
        subset = [r for r in rows if r["dataset"] == name]
        if not subset:
            continue
        won = sum(1 for r in subset if r["verdict"] == "model")
        lost = sum(1 for r in subset if r["verdict"] == "persistence")
        tied = sum(1 for r in subset if r["verdict"] == "no difference")
        mean_ratio = sum(r["mae_ratio_mean"] for r in subset) / len(subset)
        print(f"| **{name}** | {t['diff_acf1']:+.3f} | {mean_ratio:.3f} | "
              f"{won} | {lost} | {tied} |")

    corr = study.get("diff_acf1_vs_mae_ratio_correlation")
    if corr is not None:
        print(f"\nCorrelation between a trace's `diff_acf1` and its mean MAE ratio "
              f"across all 12 of its cells: **{corr:+.3f}** "
              f"(n = {len(traces)} workloads).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
