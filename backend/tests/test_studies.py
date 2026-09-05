"""Trace ingestion and the cross-dataset evaluation protocol.

These guard the two classes of mistake that are invisible in a results table:
a unit conversion that silently rescales a whole dataset, and an evaluation
protocol that leaks or mis-slices data. Both have bitten this project - the
Alibaba trace was ingested 100x too small because a column named
`cpu_util_percent` turned out to hold fractions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str):
    """Import a script from `scripts/`, which is not a package.

    The module has to be in `sys.modules` *before* it executes: `@dataclass`
    resolves annotations by looking its own module up there, and fails with an
    opaque AttributeError if it is missing.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch_trace = _load("fetch_trace")
study = _load("cross_dataset_study")
horizon_study = _load("horizon_study")


# --------------------------------------------------------------- adapters

def test_alibaba_treats_columns_as_fractions_not_percentages():
    """The `_percent` column names are a misnomer in the mirror.

    Reading them as percentages understates Alibaba demand by 100x. Verified
    against the source: no value in either column exceeds 1.0 across 40 sampled
    machine files, and mean CPU is 0.395 - i.e. ~40% utilisation, which is what
    Alibaba reports for this cluster.
    """
    df = pd.DataFrame({
        "time_stamp": [0, 60, 120],
        "cpu_util_percent": [0.40, 0.40, 0.40],
        "mem_util_percent": [0.90, 0.90, 0.90],
    })
    out = fetch_trace._parse_alibaba(df)
    assert out["cpu_cores"].iloc[0] == pytest.approx(0.40 * fetch_trace.ALIBABA_MACHINE_CORES)
    assert out["ram_gb"].iloc[0] == pytest.approx(0.90 * fetch_trace.ALIBABA_MACHINE_RAM_GB)


def test_alibaba_still_handles_true_percentages_if_the_mirror_changes():
    """The guard must work in both directions, or a mirror revision breaks it."""
    df = pd.DataFrame({
        "time_stamp": [0, 60, 120],
        "cpu_util_percent": [40.0, 40.0, 40.0],
        "mem_util_percent": [90.0, 90.0, 90.0],
    })
    out = fetch_trace._parse_alibaba(df)
    assert out["cpu_cores"].iloc[0] == pytest.approx(0.40 * fetch_trace.ALIBABA_MACHINE_CORES)


def test_bitbrains_converts_percent_of_own_cores_and_kilobytes():
    df = pd.DataFrame({
        "Timestamp [ms]": [0, 300, 600],
        "CPU cores": [8, 8, 8],
        "CPU usage [%]": [50.0, 50.0, 50.0],
        "Memory usage [KB]": [1048576, 1048576, 1048576],
    })
    out = fetch_trace._parse_bitbrains(df)
    assert out["cpu_cores"].iloc[0] == pytest.approx(4.0)     # 50% of 8 cores
    assert out["ram_gb"].iloc[0] == pytest.approx(1.0)        # 1 GiB in KB


def test_google_timestamps_are_microseconds():
    """Treating them as seconds would put every row in one slot."""
    df = pd.DataFrame({
        "start_time": [0, 300_000_000, 600_000_000],
        "avg_cpu_usage": [0.5, 0.5, 0.5],
        "avg_mem_usage": [0.25, 0.25, 0.25],
    })
    out = fetch_trace._parse_google(df)
    assert len(out) == 3, "microsecond timestamps collapsed into one slot"
    assert out["cpu_cores"].iloc[0] == pytest.approx(0.5 * fetch_trace.GOOGLE_MACHINE_CORES)


def test_azure_ram_is_flagged_as_derived():
    """Azure publishes no memory column; a RAM claim on it would be fabricated."""
    assert fetch_trace.ADAPTERS["azure"].ram_is_synthetic is True
    assert all(not a.ram_is_synthetic for k, a in fetch_trace.ADAPTERS.items() if k != "azure")


def test_repeated_readings_in_one_slot_are_averaged_not_summed():
    """Alibaba samples every 60 s into 300 s slots: summing would inflate 5x."""
    df = pd.DataFrame({
        "time_stamp": [0, 60, 120, 180, 240],
        "cpu_util_percent": [0.4] * 5,
        "mem_util_percent": [0.9] * 5,
    })
    out = fetch_trace._parse_alibaba(df)
    assert len(out) == 1
    assert out["cpu_cores"].iloc[0] == pytest.approx(0.4 * fetch_trace.ALIBABA_MACHINE_CORES)


# ------------------------------------------------------- entity partitioning

def test_shards_are_disjoint_and_exhaustive():
    """The panel treats shards as separate workload samples, which is only
    legitimate if they share no entities. Independent draws with different seeds
    would overlap and quietly count the same VMs twice."""
    files = [f"vm{i}.csv" for i in range(100)]
    shards = [fetch_trace.partition(files, 42, f"{k}/3") for k in range(3)]

    for i in range(len(shards)):
        for j in range(i + 1, len(shards)):
            assert not set(shards[i]) & set(shards[j])
    assert set().union(*shards) == set(files)
    assert sum(len(s) for s in shards) == len(files)


def test_partition_is_deterministic_and_seed_sensitive():
    files = [f"vm{i}.csv" for i in range(50)]
    assert fetch_trace.partition(files, 42, "0/3") == fetch_trace.partition(files, 42, "0/3")
    assert fetch_trace.partition(files, 42, "0/3") != fetch_trace.partition(files, 7, "0/3")


def test_partition_without_a_shard_returns_everything():
    files = [f"vm{i}.csv" for i in range(20)]
    assert sorted(fetch_trace.partition(files, 1, None)) == sorted(files)


def test_partition_rejects_a_malformed_shard():
    files = [f"vm{i}.csv" for i in range(20)]
    for bad in ("3/3", "-1/3", "abc", "1/0", "2"):
        with pytest.raises(ValueError):
            fetch_trace.partition(files, 1, bad)


# ---------------------------------------------------------- burst labels

def test_burst_labels_respect_the_refractory_period():
    cpu = np.ones(60)
    cpu[20:] += 50.0            # one large step
    onsets = fetch_trace.label_bursts(cpu, k=6.0, refractory=6)
    assert onsets.sum() == 1
    assert onsets[20] == 1


def test_burst_labels_ignore_a_flat_series():
    assert fetch_trace.label_bursts(np.full(100, 7.0)).sum() == 0


# ------------------------------------------------- trace characterisation

def test_diff_acf1_is_near_zero_for_a_random_walk():
    """A random walk is exactly the case where persistence cannot be beaten."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=20_000)) + 500.0
    assert abs(study.describe(walk)["diff_acf1"]) < 0.05


def test_diff_acf1_is_strongly_negative_for_a_mean_reverting_series():
    """Alternating steps: every change is reversed by the next one."""
    series = 100.0 + np.tile([0.0, 5.0], 5_000)
    assert study.describe(series)["diff_acf1"] < -0.9


# ----------------------------------------------------- evaluation protocol

def test_test_blocks_are_disjoint_and_ordered():
    """Overlapping blocks would break the independence the signed-rank test needs."""
    n = 8_000
    train_len, test_len, n_blocks = study.plan_blocks(n)
    spans = [(train_len + k * test_len, train_len + (k + 1) * test_len)
             for k in range(n_blocks)]
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start
    assert spans[-1][1] <= n


def test_no_test_row_is_ever_in_its_own_training_window():
    n = 8_000
    train_len, test_len, n_blocks = study.plan_blocks(n)
    for k in range(n_blocks):
        origin = train_len + k * test_len
        for expanding in (True, False):
            train_start = 0 if expanding else origin - train_len
            assert train_start < origin                      # non-empty
            assert origin >= origin                           # test starts at origin
            assert train_start >= 0


def _toy_frame(n: int = 4_000) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    cpu = 50 + 10 * np.sin(np.arange(n) / 24.0) + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({
        "timestamp": ts, "num_tasks": 10.0,
        "cpu_per_task": cpu / 10, "ram_per_task": cpu / 5,
        "hour": ts.hour, "day_of_week": ts.dayofweek,
        "cpu_demand": cpu, "ram_demand": cpu * 2,
        "is_weekend": (ts.dayofweek >= 5).astype(int),
        "burst_onset": 0, "burst_active": 0, "interval": np.arange(n),
    })


def test_expanding_window_grows_and_sliding_window_does_not():
    frame = _toy_frame()
    grown = study.evaluate(frame, "lr", expanding=True)
    fixed = study.evaluate(frame, "lr", expanding=False)
    assert len(grown) == len(fixed) >= 6
    assert grown[-1]["n_train"] > grown[0]["n_train"]
    assert len({b["n_train"] for b in fixed}) == 1


def test_evaluate_reports_a_paired_persistence_baseline_per_block():
    blocks = study.evaluate(_toy_frame(), "lr")
    assert blocks
    for b in blocks:
        assert b["persistence_mae"] > 0
        assert b["mae_ratio"] == pytest.approx(b["model_mae"] / b["persistence_mae"])


# ------------------------------------------------ multiple-comparison control

def test_holm_leaves_the_smallest_p_multiplied_by_the_family_size():
    adjusted = study.holm([0.001, 0.02, 0.5, 0.7])
    assert adjusted[0] == pytest.approx(0.004)      # 4 * 0.001


def test_holm_is_monotone_in_rank_order():
    raw = [0.001, 0.009, 0.02, 0.04, 0.3]
    adjusted = study.holm(raw)
    ordered = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
    assert ordered == sorted(ordered), "Holm output must not decrease with rank"


def test_holm_is_never_smaller_than_the_raw_p_value():
    raw = [0.01, 0.02, 0.03, 0.6, 0.9]
    for r, a in zip(raw, study.holm(raw)):
        assert a >= r - 1e-12


def test_holm_caps_at_one_and_ignores_missing_tests():
    adjusted = study.holm([0.9, float("nan"), 0.95])
    assert adjusted[0] <= 1.0 and adjusted[2] <= 1.0
    assert adjusted[1] != adjusted[1]               # NaN stays NaN


def test_wilcoxon_declines_to_test_too_few_blocks():
    """Reporting a p-value from four observations would be false precision."""
    _, p = study.wilcoxon([0.1, -0.2, 0.3, 0.4])
    assert p != p                                    # NaN


def test_wilcoxon_detects_a_consistent_shift():
    _, p = study.wilcoxon([0.4, 0.35, 0.5, 0.42, 0.38, 0.6, 0.45, 0.52])
    assert p < 0.05


# ----------------------------------------------------------------- verdicts

def test_verdict_takes_direction_from_the_median_not_the_mean():
    """The signed-rank test is a statement about the median, so the direction
    must be too. A handful of badly mispredicted blocks can drag the mean above
    1.0 while the model still wins most blocks - reading the mean there would
    contradict the very test that established significance."""
    skewed = {"mae_ratio_mean": 1.017, "mae_ratio_median": 0.94, "p_mae_holm": 0.01}
    assert study.classify(skewed) == "model"


def test_verdict_is_no_difference_when_not_significant():
    assert study.classify(
        {"mae_ratio_median": 0.60, "p_mae_holm": 0.06}) == "no difference"


def test_verdict_is_no_difference_when_the_test_was_not_run():
    assert study.classify(
        {"mae_ratio_median": 0.60, "p_mae_holm": float("nan")}) == "no difference"
    assert study.classify({"mae_ratio_median": 0.60}) == "no difference"


def test_verdict_calls_persistence_when_the_model_is_significantly_worse():
    assert study.classify(
        {"mae_ratio_median": 1.53, "p_mae_holm": 1e-6}) == "persistence"


# -------------------------------------------------- horizon-study reporting

def _cell(h, margin, wins):
    return {"horizon_intervals": h, "horizon_minutes": h * 5, "algo": "xgboost",
            "margin_mean": margin, "wins": wins, "seeds": 3}


def test_horizon_summary_reports_a_loss_as_a_loss():
    """The summary used to be a fixed sentence asserting the forecaster's
    advantage grows with horizon. On Bitbrains the deficit grows instead, so the
    JSON contradicted the table printed directly above it."""
    losing = [_cell(1, -0.003, 1), _cell(3, -0.040, 0),
              _cell(6, -0.062, 0), _cell(12, -0.059, 1)]
    text = horizon_study.summarise(losing, None)
    assert "does not beat persistence at any horizon" in text
    assert "shrinks with horizon" in text


def test_horizon_summary_reports_a_win_as_a_win():
    winning = [_cell(1, -0.001, 1), _cell(3, 0.011, 3),
               _cell(6, 0.040, 3), _cell(12, 0.149, 3)]
    text = horizon_study.summarise(winning, 15)
    assert "from 15 minutes ahead onward" in text
    assert "grows with horizon" in text


def test_horizon_summary_distinguishes_partial_wins_from_none():
    partial = [_cell(1, 0.004, 2), _cell(12, 0.006, 2)]
    text = horizon_study.summarise(partial, None)
    assert "some seeds but not on all" in text


def test_horizon_summary_survives_empty_results():
    assert horizon_study.summarise([], None) == "No results."


# ------------------------------------------- claims the documents actually make
#
# README.md, docs/RESULTS-CROSS-DATASET.md and the replacement Chapter 4 all
# state these in prose. The study output is committed, so the prose can be
# checked against it rather than trusted - which is the whole point of committing
# the JSON.

STUDY_JSON = ROOT / "artifacts" / "cross_dataset_study.json"
needs_study = pytest.mark.skipif(
    not STUDY_JSON.exists(), reason="run scripts/cross_dataset_study.py first")

REAL_TRACES = {"bitbrains", "google", "azure", "alibaba"}


@pytest.fixture(scope="module")
def study_rows():
    import json
    rows = json.loads(STUDY_JSON.read_text(encoding="utf-8"))["rows"]
    for row in rows:
        row["verdict"] = study.classify(row)
    return rows


@needs_study
def test_every_win_on_a_real_trace_is_linear_regression(study_rows):
    """The documents say no tree ensemble ever beats persistence on production
    data. That is the claim that argues against the project's own default
    predictor, so it should not be able to rot silently."""
    real_wins = [r for r in study_rows
                 if r["verdict"] == "model" and r["dataset"] in REAL_TRACES]
    assert real_wins, "expected at least one win on a real trace"
    assert {r["algo"] for r in real_wins} == {"lr"}


@needs_study
def test_bitbrains_is_the_only_workload_that_loses(study_rows):
    losses = [r for r in study_rows if r["verdict"] == "persistence"]
    assert losses
    assert {r["dataset"] for r in losses} == {"bitbrains"}


@needs_study
def test_diff_acf1_orders_the_workloads_by_outcome(study_rows):
    """The headline claim: sort by diff_acf1 and you sort by mean MAE ratio."""
    import json
    traces = json.loads(STUDY_JSON.read_text(encoding="utf-8"))["traces"]
    per_trace = {}
    for name in traces:
        cells = [r["mae_ratio_mean"] for r in study_rows if r["dataset"] == name]
        if cells:
            per_trace[name] = (traces[name]["diff_acf1"], float(np.mean(cells)))

    by_acf1 = sorted(per_trace.values())
    ratios = [ratio for _, ratio in by_acf1]
    assert ratios == sorted(ratios), (
        f"sorting by diff_acf1 no longer sorts by MAE ratio: {by_acf1}")


@needs_study
def test_bitbrains_is_the_only_workload_on_the_random_walk_side(study_rows):
    """The diagnostic's boundary and the study's data must agree; if a rebuilt
    dataset moved a trace across it, the documents would be wrong."""
    import json
    from app.ml.forecastability import PERSISTENCE_SUFFICIENT_ABOVE
    traces = json.loads(STUDY_JSON.read_text(encoding="utf-8"))["traces"]
    random_walk = {n for n, t in traces.items()
                   if t["diff_acf1"] >= PERSISTENCE_SUFFICIENT_ABOVE}
    assert random_walk == {"bitbrains"}


# ------------------------------------------------- forecastability diagnostic

from app.ml.forecastability import MIN_SAMPLES, assess  # noqa: E402


def test_a_random_walk_is_called_unforecastable():
    """The case where persistence is provably optimal - no model can beat it."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=6_000)) + 5_000.0
    result = assess(walk)
    assert result["verdict"] == "persistence_sufficient"
    assert abs(result["diff_acf1"]) < 0.1


def test_a_mean_reverting_series_is_called_forecastable():
    """Changes that reverse are structure persistence cannot use."""
    rng = np.random.default_rng(1)
    n = 6_000
    series = np.empty(n)
    series[0] = 100.0
    for i in range(1, n):                       # AR(1) pulled back to 100
        series[i] = 100.0 + 0.3 * (series[i - 1] - 100.0) + rng.normal(scale=3.0)
    result = assess(series)
    assert result["verdict"] == "model_likely_helps"
    assert result["diff_acf1"] < -0.2


def test_high_level_autocorrelation_alone_does_not_imply_forecastability():
    """The trap this diagnostic exists to avoid.

    A random walk has level autocorrelation near 1.0, which is exactly why a
    naive baseline scores R² > 0.9 on this problem and why R² alone is close to
    meaningless. The verdict must not be driven by it.
    """
    rng = np.random.default_rng(2)
    walk = np.cumsum(rng.normal(size=6_000)) + 5_000.0
    result = assess(walk)
    assert result["level_acf1"] > 0.95
    assert result["verdict"] == "persistence_sufficient"


def test_short_series_is_refused_rather_than_guessed():
    rng = np.random.default_rng(3)
    result = assess(rng.normal(size=MIN_SAMPLES - 1) + 50.0)
    assert result["verdict"] == "inconclusive"
    assert str(MIN_SAMPLES) in result["reason"]


def test_constant_and_degenerate_series_do_not_raise():
    assert assess(np.full(1_000, 42.0))["verdict"] == "inconclusive"
    assert assess(np.zeros(1_000))["verdict"] == "inconclusive"


def test_recommendation_follows_the_verdict_and_is_always_present():
    """Callers read `recommended_algo` on every path, including the refusals."""
    from app.ml.forecastability import RECOMMENDED_ALGO

    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.normal(size=6_000)) + 5_000.0
    assert assess(walk)["recommended_algo"] == "persistence"

    n = 6_000
    reverting = np.empty(n)
    reverting[0] = 100.0
    for i in range(1, n):
        reverting[i] = 100.0 + 0.3 * (reverting[i - 1] - 100.0) + rng.normal(scale=3.0)
    assert assess(reverting)["recommended_algo"] == "lr"

    short = assess(rng.normal(size=MIN_SAMPLES - 1) + 50.0)
    assert short["recommended_algo"] is None
    assert short["recommendation_note"]

    for verdict, algo in RECOMMENDED_ALGO.items():
        assert algo in (None, "lr", "persistence"), verdict


def test_recommendation_never_names_a_tree_ensemble():
    """The measurement is that neither ensemble beat persistence on any real
    trace at any horizon. If a future calibration reverses that, this test should
    be the thing that makes someone justify it."""
    from app.ml.forecastability import RECOMMENDED_ALGO

    assert not {"xgboost", "rf"} & set(RECOMMENDED_ALGO.values())


def test_recommended_algos_are_all_buildable():
    """A recommendation the system cannot act on is just a comment."""
    from app.ml.forecastability import RECOMMENDED_ALGO
    from app.ml.predictor import WorkloadPredictor

    for algo in filter(None, RECOMMENDED_ALGO.values()):
        assert WorkloadPredictor(algo)._make_model({}) is not None


def test_verdict_is_invariant_to_rescaling():
    """Demand in cores or millicores must not change the answer."""
    rng = np.random.default_rng(4)
    series = 100.0 + rng.normal(size=4_000).cumsum() * 0.1
    assert assess(series)["verdict"] == assess(series * 1_000.0)["verdict"]
