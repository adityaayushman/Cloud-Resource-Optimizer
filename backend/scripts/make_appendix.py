"""Generate Appendix A (source listing) from the code that actually runs.

The submitted appendix does not run and is not this project's code. It contains a
`load_and_process_data()` that parses `id`/`cycle`/`s1..s21` columns and computes
Remaining Useful Life from a turbofan dataset - a different domain entirely. What
of it does belong here is broken: `engine.py` imports a `models` module the
appendix never lists, `predict()` passes a raw NumPy array to models fitted on a
named DataFrame, and `fillna(method='bfill')` was removed in pandas 3.

This emits the listing from the working source, in the module order the report's
architecture table uses, and **imports every module first** so a listing that
cannot execute fails here rather than in a viva.

    python scripts/make_appendix.py --modules core   # the algorithmic content
    python scripts/make_appendix.py --modules all    # everything, ~4,000 lines

Output is Markdown with fenced code blocks: paste into Word, or run it through
pandoc for LaTeX.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Order follows the architecture table in the README: domain types first, then
# the catalogue they are priced against, the workload they are exercised by, the
# learning components, and finally the code that composes them.
CORE = [
    ("app/models.py", "Domain types - instance classes, VM and task records"),
    ("app/catalog.py", "Instance catalogue, provider pricing, multi-cloud scoring"),
    ("app/workload.py", "Synthetic workload generator and production-trace replay"),
    ("app/ml/predictor.py", "Demand forecasting, causal features, TreeSHAP, persistence"),
    ("app/ml/dqn.py", "Deep Q-Network in NumPy - manual backprop, Adam, target network"),
    ("app/ml/anomaly.py", "Isolation Forest and z-score detectors, event-based scoring"),
    ("app/ml/forecastability.py", "Pre-training diagnostic: is a forecaster worth building?"),
]

EXTRA = [
    ("app/ml/qlearning.py", "Tabular Q-learning, the comparison arm"),
    ("app/engine.py", "Allocator, autoscaler, advisory engine, RL glue"),
    ("app/simulation.py", "Closed-loop harness and the controlled ablation"),
    ("app/main.py", "FastAPI application"),
]


def verify_imports(paths: list[str]) -> list[str]:
    """Import each module. A listing that cannot execute is worse than none."""
    failures = []
    for path in paths:
        module = path.replace("/", ".").removesuffix(".py")
        try:
            importlib.import_module(module)
        except Exception as exc:                      # noqa: BLE001 - report any
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", choices=("core", "all"), default="core")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "docs" / "APPENDIX-A.md")
    args = ap.parse_args()

    listing = CORE if args.modules == "core" else CORE + EXTRA

    print(f"Verifying {len(listing)} modules import cleanly...")
    failures = verify_imports([p for p, _ in listing])
    if failures:
        print("ERROR: the listing would not run:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("  all import cleanly")

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    total = sum(len((ROOT / p).read_text(encoding="utf-8").splitlines())
                for p, _ in listing)

    parts = [
        "# Appendix A — Source Listing",
        "",
        f"Generated from commit `{commit}` by `backend/scripts/make_appendix.py`.",
        "Every module below was imported successfully before this document was "
        "written, so the listing is executable code rather than a transcription.",
        "",
        f"{len(listing)} modules, {total:,} lines. The full repository, including "
        "the test suite and the evaluation scripts, is at "
        "`github.com/adityaayushman/Cloud-Resource-Optimizer`.",
        "",
        "## Contents",
        "",
    ]
    for i, (path, blurb) in enumerate(listing, start=1):
        parts.append(f"A.{i}  `{path}` — {blurb}")
        parts.append("")

    for i, (path, blurb) in enumerate(listing, start=1):
        source = (ROOT / path).read_text(encoding="utf-8").rstrip()
        lines = len(source.splitlines())
        parts += [
            "---", "",
            f"## A.{i}  `{path}`", "",
            f"*{blurb}. {lines} lines.*", "",
            "```python", source, "```", "",
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {args.out}  ({total:,} lines of source, "
          f"{len('\n'.join(parts).splitlines()):,} lines of document)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
