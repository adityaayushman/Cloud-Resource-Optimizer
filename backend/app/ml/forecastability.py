"""Is a learned forecaster worth building for *this* workload?

Most of this project's ML machinery is justified by a benchmark table. Those
tables turned out to be workload-specific: on the synthetic generator a learned
forecaster beats a persistence baseline from 15 minutes out, on the Bitbrains
production trace it never beats it at any horizon, and on Google, Azure and
Alibaba it beats it decisively. Three contradictory conclusions, three datasets,
same code.

`scripts/cross_dataset_study.py` measured all five and found the dependence is
not arbitrary. It is predicted by one statistic that costs a single pass over the
trace, needs no training, and can be computed before any model exists:

    diff_acf1 = corr(d[t], d[t+1])    where   d[t] = cpu[t+1] - cpu[t]

the lag-1 autocorrelation of the **first difference**.

    diff_acf1 ~ 0    the series is a random walk. This interval's change says
                     nothing about the next one, so "next equals current" is
                     already the best available forecast. Nothing can beat it,
                     and a model that tries will fit noise and do worse.

    diff_acf1 << 0   changes reverse: demand overshoots and settles back.
                     Persistence structurally cannot exploit that - it always
                     predicts the overshoot continues. A model can.

Note what is *not* useful here. The autocorrelation of the level (`acf1`) is
above 0.84 on all five workloads including the random walk, because all cloud
demand is smooth at five-minute resolution. A high `acf1` is why a naive
baseline scores R² > 0.9 and why R² alone is close to meaningless on this
problem. It says nothing about whether a model can add value.

There is a suggestive second reading, offered here as a caution rather than a
result. A reactive threshold autoscaler scales to the *last* observation, so it
is a persistence forecaster wearing a different hat, and on the most strongly
mean-reverting workload measured (Google, diff_acf1 = -0.52) the reactive arm did
not merely underperform - it rejected 67.6% of all work. But across the four
workloads that arm was run on, the correlation between diff_acf1 and its failure
rate is only -0.465, and the second-most mean-reverting workload had the *lowest*
failure rate of the four. The mechanism is plausible; four points do not
establish it.

See `docs/RESULTS-CROSS-DATASET.md` for the measurements behind the thresholds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Boundaries calibrated on the five workloads in the cross-dataset study. They
# are a decision aid, not a law: five points establish that the dependence
# exists and roughly where it turns, not a universal constant. The band between
# them is deliberately wide, and anything inside it is reported as inconclusive
# rather than guessed.
PERSISTENCE_SUFFICIENT_ABOVE = -0.10
MODEL_LIKELY_HELPS_BELOW = -0.20

# Below this many samples the statistic is too noisy to act on. 288 five-minute
# samples is one day.
MIN_SAMPLES = 288


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def assess(cpu_demand: Sequence[float], interval_minutes: int = 5) -> dict:
    """Judge, without training anything, whether a learned forecaster will pay.

    Returns the diagnostic statistics, a verdict, and the reasoning behind it.
    `verdict` is one of:

        "model_likely_helps"      demand mean-reverts; a forecaster has
                                  structure to exploit that persistence cannot
        "persistence_sufficient"  demand is close to a random walk; ship the
                                  two-line baseline instead
        "inconclusive"            between the two, or too little data
    """
    series = np.asarray(cpu_demand, dtype=float)
    series = series[np.isfinite(series)]

    if len(series) < MIN_SAMPLES:
        return {
            "verdict": "inconclusive",
            "reason": f"only {len(series)} samples; at least {MIN_SAMPLES} "
                      f"({MIN_SAMPLES * interval_minutes / 60:.0f} h) are needed "
                      f"before this statistic is stable",
            "samples": int(len(series)),
        }

    mean = float(series.mean())
    if mean <= 0:
        return {"verdict": "inconclusive", "reason": "demand is zero or negative",
                "samples": int(len(series))}

    diff = np.diff(series)
    diff_acf1 = _autocorr(diff, 1)
    if not np.isfinite(diff_acf1):
        return {"verdict": "inconclusive", "reason": "demand is constant",
                "samples": int(len(series))}

    if diff_acf1 <= MODEL_LIKELY_HELPS_BELOW:
        verdict = "model_likely_helps"
        reason = (
            f"Changes in demand reverse (diff_acf1 = {diff_acf1:+.3f}): a rise is "
            f"typically followed by a fall. A persistence baseline always predicts "
            f"the rise continues, so it is systematically wrong in a way a learned "
            f"model can correct. A reactive threshold autoscaler scales to the last "
            f"observation and so shares that blind spot; worth testing before "
            f"relying on one here."
        )
    elif diff_acf1 >= PERSISTENCE_SUFFICIENT_ABOVE:
        verdict = "persistence_sufficient"
        reason = (
            f"Demand behaves like a random walk (diff_acf1 = {diff_acf1:+.3f}): this "
            f"interval's change carries no information about the next. 'Next "
            f"interval equals this one' is already close to the best available "
            f"forecast, and a learned model will mostly fit noise. Spend the "
            f"complexity budget on the control policy instead."
        )
    else:
        verdict = "inconclusive"
        reason = (
            f"diff_acf1 = {diff_acf1:+.3f} falls between the calibrated boundaries "
            f"({MODEL_LIKELY_HELPS_BELOW:+.2f} and {PERSISTENCE_SUFFICIENT_ABOVE:+.2f}). "
            f"Measure it directly: train one model and compare MAE against the "
            f"persistence baseline on held-out blocks."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "diff_acf1": round(diff_acf1, 4),
        "level_acf1": round(_autocorr(series, 1), 4),
        "level_acf12": round(_autocorr(series, 12), 4),
        "cv": round(float(series.std() / mean), 4),
        "mean_abs_step_pct": round(float(np.abs(diff).mean() / mean * 100), 3),
        "samples": int(len(series)),
        "span_hours": round(len(series) * interval_minutes / 60.0, 1),
        "thresholds": {
            "model_likely_helps_below": MODEL_LIKELY_HELPS_BELOW,
            "persistence_sufficient_above": PERSISTENCE_SUFFICIENT_ABOVE,
        },
        "caveat": (
            "Calibrated on five workloads (one synthetic, four production traces). "
            "It is a screening heuristic, not a proof - it says where to spend "
            "measurement effort, not what the answer will be."
        ),
    }
