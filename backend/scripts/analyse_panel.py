"""How strong is the diff_acf1 relationship, once clustering is accounted for?

The five-workload study reported r = +0.956 over five points. The panel widens
that to seventeen samples, but seventeen samples are **not** seventeen
independent workloads and reporting them as such would be a worse error than the
small sample it fixes: three shards of Bitbrains fastStorage share a datacentre,
a month and a diurnal cycle, so their agreement is close to guaranteed and adds
much less information than three separate traces would.

This reports the correlation at three levels of aggregation, from most powerful
and most correlated to most conservative:

    sample     all 17 panel members. Highest n, strongly clustered, so its
               p-value is optimistic and is labelled as such.

    collection distinct source-and-period: Bitbrains fastStorage, the three
               Bitbrains Rnd months, Google, Azure, Alibaba, synthetic. Shards
               within a collection are averaged first. These are separate
               realisations - different machines, and for the Rnd months
               different months - which is the level the claim is really made at.

    provider   one point per provider plus synthetic. Most defensible, least
               powerful, and back to roughly the n the original study had.

**All p-values here are permutation p-values, enumerated exactly whenever the
sample is small enough to allow it.** scipy's parametric ones cannot be trusted
at this n - `spearmanr` reports p = 1e-24 for a perfect rank correlation over
five points, where the exact answer is 2/120 = 0.017, because no arrangement
other than the observed one and its reverse achieves it.

    python scripts/analyse_panel.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _study_module():
    spec = importlib.util.spec_from_file_location(
        "cross_dataset_study", Path(__file__).parent / "cross_dataset_study.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cross_dataset_study"] = module
    spec.loader.exec_module(module)
    return module


classify = _study_module().classify


def collection_of(name: str) -> str:
    """Distinct source *and* period - the level the claim is really made at."""
    if name.startswith("bb_fs"):
        return "bitbrains/fastStorage"
    if name.startswith("bb_rnd_"):
        return f"bitbrains/Rnd-{name.split('_')[-1]}"
    if name.startswith("syn_"):
        return "synthetic"
    return name.rsplit("_", 1)[0]


def provider_of(name: str) -> str:
    if name.startswith("bb_"):
        return "bitbrains"
    if name.startswith("syn_"):
        return "synthetic"
    return name.rsplit("_", 1)[0]


MAX_EXACT_PERMUTATIONS = 40_320          # 8!


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.corrcoef(xs, ys)[0, 1])


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    from scipy.stats import rankdata
    return float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])


def permutation_p(xs: np.ndarray, ys: np.ndarray, trials: int = 20_000,
                  statistic=_pearson) -> float:
    """P(|r| this large by chance) with no normality assumption.

    Two things this has to get right, both of which it originally got wrong.

    The observed statistic must be computed with the *same* function used inside
    the loop. Taking it from `scipy.stats.pearsonr` and comparing against
    `np.corrcoef` differs in the last bits, so the identity permutation failed its
    own `>=` test and the p-value came out two orders of magnitude too small
    (5e-5 where the true value was 0.0083).

    And at these sample sizes the permutation distribution is tiny - five points
    admit only 120 arrangements, so no sampling scheme can report a p below
    1/120. Small n is therefore enumerated exactly rather than sampled, which
    also removes the floor artefact entirely.
    """
    import itertools
    import math

    n = len(xs)
    observed = abs(statistic(xs, ys))
    tol = 1e-12

    if math.factorial(n) <= MAX_EXACT_PERMUTATIONS:
        hits = sum(1 for perm in itertools.permutations(range(n))
                   if abs(statistic(xs, ys[list(perm)])) >= observed - tol)
        return hits / math.factorial(n)

    rng = np.random.default_rng(0)
    shuffled = ys.copy()
    hits = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if abs(statistic(xs, shuffled)) >= observed - tol:
            hits += 1
    return (hits + 1) / (trials + 1)


def correlate(pairs: list[tuple[float, float]], label: str, optimistic: bool = False):
    from scipy.stats import pearsonr, spearmanr

    xs = np.array([a for a, _ in pairs])
    ys = np.array([b for _, b in pairs])
    n = len(xs)
    if n < 3:
        print(f"{label:<28} n={n:<3} too few to correlate")
        return None

    r, _ = pearsonr(xs, ys)
    rho, _ = spearmanr(xs, ys)

    # scipy's parametric p-values lean on asymptotics that do not hold at this n -
    # it reports p = 1e-24 for a perfect rank correlation over five points, where
    # the exact answer is 2/120 = 0.017. Permutation p-values are used throughout
    # instead, enumerated exactly whenever n! is small enough.
    p_r = permutation_p(xs, ys, statistic=_pearson)
    p_rho = permutation_p(xs, ys, statistic=_spearman)

    exact = __import__("math").factorial(n) <= MAX_EXACT_PERMUTATIONS
    kind = "exact" if exact else "sampled"
    flag = "  (clustered - optimistic)" if optimistic else ""
    print(f"{label:<28} n={n:<3} r={r:+.3f} p={p_r:.3g}   "
          f"rho={rho:+.3f} p={p_rho:.3g}   [{kind} permutation]{flag}")
    return {"level": label, "n": n, "pearson_r": round(float(r), 4),
            "pearson_permutation_p": float(p_r),
            "spearman_rho": round(float(rho), 4),
            "spearman_permutation_p": float(p_rho),
            "p_values": f"{kind} permutation, not parametric",
            "clustered": optimistic}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path,
                    default=ROOT / "artifacts" / "panel_study.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "panel_analysis.json")
    args = ap.parse_args()

    study = json.loads(args.json.read_text(encoding="utf-8"))
    rows = study["rows"]
    for row in rows:
        row["verdict"] = classify(row)
    traces = study["traces"]

    per_sample = {}
    for name in traces:
        cells = [r for r in rows if r["dataset"] == name]
        if not cells:
            continue
        per_sample[name] = {
            "diff_acf1": traces[name]["diff_acf1"],
            "mae_ratio": float(np.mean([c["mae_ratio_mean"] for c in cells])),
            "won": sum(1 for c in cells if c["verdict"] == "model"),
            "lost": sum(1 for c in cells if c["verdict"] == "persistence"),
            "cells": len(cells),
        }

    print(f"{'sample':<16}{'diff_acf1':>11}{'mean MAE ratio':>16}{'won':>6}{'lost':>6}")
    print("-" * 55)
    for name, v in sorted(per_sample.items(), key=lambda kv: kv[1]["diff_acf1"]):
        print(f"{name:<16}{v['diff_acf1']:>+11.3f}{v['mae_ratio']:>16.3f}"
              f"{v['won']:>6}{v['lost']:>6}")

    # --- reliability: do shards of one collection agree with each other? ------
    groups = defaultdict(list)
    for name, v in per_sample.items():
        groups[collection_of(name)].append(v)
    spreads = [(np.ptp([v["diff_acf1"] for v in g]),
                np.ptp([v["mae_ratio"] for v in g]))
               for g in groups.values() if len(g) > 1]
    print()
    if spreads:
        print(f"Within-collection spread across shards (max-min), "
              f"{len(spreads)} collections with >1 shard:")
        print(f"  diff_acf1      {np.mean([s[0] for s in spreads]):.3f} mean, "
              f"{max(s[0] for s in spreads):.3f} worst")
        print(f"  mean MAE ratio {np.mean([s[1] for s in spreads]):.3f} mean, "
              f"{max(s[1] for s in spreads):.3f} worst")
        print("  Small spreads mean the shards agree - which is reassurance about "
              "measurement, not extra evidence for the claim.")

    # --- correlation at three levels of aggregation --------------------------
    print()
    levels = []
    levels.append(correlate(
        [(v["diff_acf1"], v["mae_ratio"]) for v in per_sample.values()],
        "sample (all panel members)", optimistic=True))

    for label, keyfn in (("collection (source+period)", collection_of),
                         ("provider (most conservative)", provider_of)):
        buckets = defaultdict(list)
        for name, v in per_sample.items():
            buckets[keyfn(name)].append(v)
        pairs = [(float(np.mean([v["diff_acf1"] for v in g])),
                  float(np.mean([v["mae_ratio"] for v in g])))
                 for g in buckets.values()]
        levels.append(correlate(pairs, label))

    payload = {
        "source": str(args.json.name),
        "per_sample": per_sample,
        "levels": [x for x in levels if x],
        "note": ("Seventeen panel members are not seventeen independent workloads. "
                 "Shards of one collection share a datacentre, a month and a "
                 "diurnal cycle, so the sample-level p-value is optimistic. The "
                 "collection level is the one the claim should be read at."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
