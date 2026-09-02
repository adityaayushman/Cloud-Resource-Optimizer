"""Run the controlled ablation study and write `artifacts/ablation.json`.

Every arm sees an identical workload trace (same generator, same seed), so the
only variable is the control policy. Results are measured, not asserted.

    python scripts/evaluate.py --ticks 288 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.simulation import STRATEGY_LABELS, SimulationHarness  # noqa: E402

ARTIFACTS = ROOT / "artifacts"

METRIC_KEYS = [
    "utilisation", "cost_per_day", "response_latency_s",
    "task_failure_rate", "sla_compliance", "co2_kg", "mean_fleet_size",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=288, help="288 ticks x 5 min = 24 h")
    ap.add_argument("--repeats", type=int, default=3, help="seeds per arm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trace", type=Path, default=None,
                    help="replay a recorded trace instead of the synthetic generator")
    ap.add_argument("--artifacts", type=Path, default=None,
                    help="model directory (default: backend/artifacts)")
    ap.add_argument("--max-fleet", type=int, default=40)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    global ARTIFACTS
    if args.artifacts:
        ARTIFACTS = args.artifacts
    out_path = args.out or (ARTIFACTS / "ablation.json")

    harness = SimulationHarness(
        artifacts_dir=ARTIFACTS,
        trace_path=str(args.trace) if args.trace else None,
        max_fleet=args.max_fleet,
    )
    started = time.time()

    kind = f"real trace ({args.trace.name})" if args.trace else "synthetic generator"
    print(f"Ablation on {kind}: {len(STRATEGY_LABELS)} arms x {args.repeats} seeds "
          f"x {args.ticks} ticks ({args.ticks * 5 / 60:.0f}h simulated)")
    print(f"  mean demand {harness.reference_cpu:.1f} cores, "
          f"RAM:CPU {harness.reference_ratio:.2f}, fleet cap {args.max_fleet}\n")

    per_arm: dict[str, list[dict]] = {}
    for name in STRATEGY_LABELS:
        runs = []
        for r in range(args.repeats):
            run = harness.run(name, ticks=args.ticks, seed=args.seed + r,
                              train_rl=False, greedy=True)
            runs.append(run.summary)
        per_arm[name] = runs
        util = statistics.mean(s["utilisation_mean"] for s in runs)
        cost = statistics.mean(s["cost_per_day"] for s in runs)
        lat = statistics.mean(s["response_latency_s"] for s in runs)
        print(f"  {name:<20} util={util:5.1f}%  cost=${cost:7.2f}/day  "
              f"latency={lat:6.1f}s")

    def agg(name: str, key: str, src: str) -> tuple[float, float]:
        vals = [s[src] for s in per_arm[name]]
        return (
            round(statistics.mean(vals), 3),
            round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
        )

    source = {
        "utilisation": "utilisation_mean",
        "cost_per_day": "cost_per_day",
        "response_latency_s": "response_latency_s",
        "task_failure_rate": "task_failure_rate",
        "sla_compliance": "sla_compliance",
        "co2_kg": "co2_kg",
        "mean_fleet_size": "mean_fleet_size",
    }

    baseline = "ml_predictive"
    base_vals = {k: agg(baseline, k, v)[0] for k, v in source.items()}

    rows = []
    for name, label in STRATEGY_LABELS.items():
        row = {"strategy": name, "label": label}
        for key, src in source.items():
            mean, sd = agg(name, key, src)
            row[key] = mean
            row[f"{key}_sd"] = sd
        for key in ("utilisation", "cost_per_day", "response_latency_s"):
            base = base_vals[key] or 1e-9
            row[f"delta_{key}_pct"] = round((row[key] - base) / base * 100, 1)
        rows.append(row)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workload": {
            "kind": "real_trace" if args.trace else "synthetic",
            "source": str(args.trace.name) if args.trace else "app/workload.py generator",
            "reference_cpu_mean": round(harness.reference_cpu, 2),
            "ram_cpu_ratio": round(harness.reference_ratio, 3),
            "max_fleet": args.max_fleet,
        },
        "protocol": {
            "ticks": args.ticks,
            "tick_seconds": 300,
            "simulated_hours": round(args.ticks * 300 / 3600, 2),
            "repeats": args.repeats,
            "seeds": [args.seed + r for r in range(args.repeats)],
            "baseline": baseline,
            "note": (
                "All arms run on an identical workload trace per seed. Values are "
                "means over seeds with the sample standard deviation reported "
                "alongside. Deltas are relative to the ML-prediction-only baseline."
            ),
        },
        "rows": rows,
        "raw": per_arm,
        "elapsed_seconds": round(time.time() - started, 2),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path} in {payload['elapsed_seconds']}s")
    print(f"\n{'Configuration':<38} {'Util%':>7} {'$/day':>8} {'Latency':>9} {'Fail%':>7}")
    print("-" * 74)
    for row in rows:
        print(f"{row['label'][:37]:<38} {row['utilisation']:>7.1f} "
              f"{row['cost_per_day']:>8.2f} {row['response_latency_s']:>8.1f}s "
              f"{row['task_failure_rate']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
