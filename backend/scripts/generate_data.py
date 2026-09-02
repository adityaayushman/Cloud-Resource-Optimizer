"""Generate `data/workload_history.csv`.

Schema (blueprint 5.1) plus three extra columns the report's schema did not
carry but which the anomaly evaluation needs:

    timestamp, num_tasks, cpu_per_task, ram_per_task, hour, day_of_week,
    cpu_demand, ram_demand,          <- required by the predictor
    is_weekend, burst_active, interval   <- extra, used for evaluation/labelling

`burst_active` is a ground-truth label for the injected demand spikes, which is
what lets `train.py` report the anomaly detector's recall instead of just
asserting that it works.

Usage:
    python scripts/generate_data.py --days 30 --interval 5 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.workload import WorkloadConfig, WorkloadGenerator  # noqa: E402

DEFAULT_START = "2026-01-01 00:00:00"


def build(days: int, interval_minutes: int, seed: int) -> pd.DataFrame:
    cfg = WorkloadConfig(interval_minutes=interval_minutes)
    gen = WorkloadGenerator(cfg, seed=seed)
    n = int(days * 24 * 60 / interval_minutes)
    rows = gen.generate(n)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.date_range(
        DEFAULT_START, periods=len(df), freq=f"{interval_minutes}min"
    )
    ordered = [
        "timestamp", "num_tasks", "cpu_per_task", "ram_per_task",
        "hour", "day_of_week", "cpu_demand", "ram_demand",
        "is_weekend", "burst_active", "burst_onset", "interval",
    ]
    return df[ordered]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the workload history dataset.")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--interval", type=int, default=5, help="sampling interval in minutes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "workload_history.csv",
    )
    args = ap.parse_args()

    df = build(args.days, args.interval, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"  span          : {df['timestamp'].iloc[0]}  ->  {df['timestamp'].iloc[-1]}")
    print(f"  interval      : {args.interval} min")
    print(f"  cpu_demand    : mean {df['cpu_demand'].mean():.2f}  "
          f"min {df['cpu_demand'].min():.2f}  max {df['cpu_demand'].max():.2f}")
    print(f"  ram_demand    : mean {df['ram_demand'].mean():.2f}  "
          f"min {df['ram_demand'].min():.2f}  max {df['ram_demand'].max():.2f}")
    print(f"  burst rows    : {int(df['burst_active'].sum()):,} "
          f"({df['burst_active'].mean() * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
