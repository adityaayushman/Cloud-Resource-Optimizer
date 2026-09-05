"""Predictor, DQN, Q-learning and anomaly detector behaviour.

These tests guard the properties that are easy to get silently wrong: feature
causality, split ordering, gradient correctness and target-network syncing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.anomaly import AnomalyDetector, build_features
from app.ml.dqn import ACTION_DIM, STATE_DIM, DQNAgent, DQNConfig, MLP, ReplayBuffer
from app.ml.predictor import (
    FEATURES,
    HORIZON,
    SplitMetrics,
    WorkloadPredictor,
    chronological_split,
    prepare_features,
)
from app.ml.qlearning import QLearningAgent
from app.workload import build_dataset

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.fixture(scope="module")
def frame():
    """Small fixture for property tests that do not depend on sample size."""
    return pd.DataFrame(build_dataset(days=14, seed=7, interval_minutes=5))


@pytest.fixture(scope="module")
def production_frame():
    """Same size and generator settings as the deployed dataset.

    Accuracy claims are sample-size sensitive: on ~2 weeks of 5-minute samples
    the ensembles cannot separate from the persistence baseline, and a test
    asserting otherwise would only be proving the fixture was too small. Any
    test that makes a claim about model *quality* uses this one.
    """
    return pd.DataFrame(build_dataset(days=30, seed=42, interval_minutes=5))


# ------------------------------------------------------- feature causality

def test_lag_feature_equals_previous_target(frame):
    feat = prepare_features(frame)
    # cpu_lag_1 at row i must be cpu_demand at row i-1 of the same frame.
    assert feat["cpu_lag_1"].iloc[5] == pytest.approx(feat["cpu_demand"].iloc[4])


def test_rolling_mean_excludes_the_current_value(frame):
    """A rolling window containing its own target is leakage."""
    feat = prepare_features(frame)
    i = 40
    expected = feat["cpu_demand"].iloc[i - 4:i].mean()
    assert feat["cpu_rolling_mean_4"].iloc[i] == pytest.approx(expected, rel=1e-6)


def test_target_is_the_next_interval(frame):
    feat = prepare_features(frame)
    assert feat["target_cpu"].iloc[10] == pytest.approx(feat["cpu_demand"].iloc[10 + HORIZON])


def test_no_nan_survives_feature_preparation(frame):
    feat = prepare_features(frame)
    assert not feat[FEATURES + ["target_cpu", "target_ram"]].isna().any().any()


def test_split_is_chronological_and_disjoint(frame):
    feat = prepare_features(frame)
    train, val, test = chronological_split(feat)
    assert len(train) + len(val) + len(test) == len(feat)
    # Every training row precedes every validation row, and so on.
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


# -------------------------------------------------------------- predictor

def test_predictor_is_competitive_with_persistence_at_one_step(production_frame):
    """At a single 5-minute step the series is autocorrelated enough that
    persistence is a very strong baseline - measured across seeds, XGBoost beats
    it only marginally and not always. The property that must hold is that the
    model is *competitive*, not that it always wins; claiming a decisive win at
    this horizon would not survive a re-run on a different seed.

    The horizon at which the model does separate is measured by
    `scripts/horizon_study.py` and asserted below.
    """
    report = WorkloadPredictor("xgboost").train(production_frame, tune=False)
    cpu = report["targets"]["cpu_demand_t+1"]
    persistence = cpu["naive_persistence_test"]["r2"]
    assert cpu["test"]["r2"] > 0.80
    assert cpu["test"]["r2"] > persistence - 0.05


def test_persistence_predictor_reproduces_the_reported_baseline(production_frame):
    """`algo="persistence"` must score exactly what the report's baseline column
    scores, or the system would be shipping something other than the thing the
    cross-dataset study recommends on random-walk workloads.

    The trap it must avoid is `cpu_lag_1`, which is demand at t-1 while the target
    is t+1 - a two-step forecast wearing the name of a one-step one.
    """
    report = WorkloadPredictor("persistence").train(production_frame, tune=False)
    cpu = report["targets"]["cpu_demand_t+1"]
    baseline = cpu["naive_persistence_test"]

    # The synthetic generator rounds num_tasks, so the reconstruction of current
    # demand is exact only up to that rounding; on the production traces it agrees
    # to 1e-13.
    assert cpu["test"]["mae"] == pytest.approx(baseline["mae"], rel=0.05)
    assert cpu["test"]["r2"] == pytest.approx(baseline["r2"], abs=0.01)


def test_persistence_predictor_does_not_use_the_stale_lag_column():
    """cpu_lag_1 is t-1. Using it would silently degrade the forecast."""
    from app.ml.predictor import PersistenceRegressor

    X = pd.DataFrame({
        "num_tasks": [10.0, 20.0], "cpu_per_task": [2.0, 3.0],
        "ram_per_task": [4.0, 5.0], "cpu_lag_1": [999.0, 999.0],
        "ram_lag_1": [888.0, 888.0],
    })
    assert list(PersistenceRegressor("cpu").fit(X).predict(X)) == [20.0, 60.0]
    assert list(PersistenceRegressor("ram").fit(X).predict(X)) == [40.0, 100.0]


def test_persistence_predictor_round_trips_and_explains(tmp_path, frame):
    predictor = WorkloadPredictor("persistence")
    predictor.train(frame, tune=False)
    predictor.save(tmp_path)

    restored = WorkloadPredictor.load(tmp_path, "persistence")
    probe = {f: 1.5 for f in restored.features}
    probe.update(num_tasks=10.0, cpu_per_task=2.0)
    assert restored.predict(probe)[0] == pytest.approx(20.0)

    explanation = restored.explain(probe)
    assert explanation["method"] == "identity-exact"
    contributing = {c["feature"] for c in explanation["contributions"]
                    if abs(c["contribution"]) > 1e-9}
    assert contributing == {"num_tasks", "cpu_per_task"}


def test_predictor_clearly_beats_persistence_at_a_longer_horizon(production_frame):
    """The forecaster earns its place once it must see past the autocorrelation.

    This is the justification for having an ML layer at all, so it is asserted
    rather than left to a report claim.
    """
    import app.ml.predictor as predictor_module

    original = predictor_module.HORIZON
    try:
        predictor_module.HORIZON = 6          # 30 minutes ahead
        report = WorkloadPredictor("xgboost").train(production_frame, tune=False)
    finally:
        predictor_module.HORIZON = original

    cpu = report["targets"]["cpu_demand_t+1"]
    margin = cpu["test"]["r2"] - cpu["naive_persistence_test"]["r2"]
    assert margin > 0.01, f"expected a clear win at 30 min ahead, got {margin:+.4f}"


# --------------------------------------------------------------------- DQN

def test_mlp_gradient_matches_numerical_estimate():
    """Hand-written backprop is the easiest thing here to get subtly wrong."""
    rng = np.random.default_rng(0)
    net = MLP([3, 8, 2], seed=1)
    X = rng.normal(size=(4, 3))
    target = rng.normal(size=(4, 2))

    out, cache = net.forward(X)
    # np.mean averages over rows *and* output units, so the derivative divides
    # by out.size, not by the batch size.
    d_out = 2 * (out - target) / out.size
    grads_W, _ = net.backward(cache, d_out)

    eps = 1e-6
    i, j = 1, 1
    original = net.W[0][i, j]

    net.W[0][i, j] = original + eps
    loss_hi = np.mean((net(X) - target) ** 2)
    net.W[0][i, j] = original - eps
    loss_lo = np.mean((net(X) - target) ** 2)
    net.W[0][i, j] = original

    numerical = (loss_hi - loss_lo) / (2 * eps)
    assert grads_W[0][i, j] == pytest.approx(numerical, rel=1e-3, abs=1e-6)


def test_agent_action_is_in_range():
    agent = DQNAgent(seed=3)
    for _ in range(30):
        a = agent.act(np.zeros(STATE_DIM))
        assert 0 <= a < ACTION_DIM


def test_target_network_is_synced_on_schedule():
    """A target network that is never synced makes every Bellman target noise."""
    agent = DQNAgent(seed=5, config=DQNConfig(warmup=8, batch_size=8, target_sync=5))
    for _ in range(200):
        agent.remember(np.random.rand(STATE_DIM), 1, 0.5, np.random.rand(STATE_DIM), False)
    for _ in range(5):
        agent.learn()
    assert agent.learn_steps == 5
    for online, target in zip(agent.online.W, agent.target.W):
        assert np.allclose(online, target), "target network did not sync at the interval"


def test_learning_reduces_loss_on_a_fixed_target():
    agent = DQNAgent(seed=11, config=DQNConfig(warmup=32, batch_size=32, target_sync=10_000))
    state = np.ones(STATE_DIM)
    for _ in range(400):
        agent.remember(state, 2, 1.0, state, True)
    first = [agent.learn() for _ in range(5)]
    for _ in range(150):
        agent.learn()
    last = [agent.learn() for _ in range(5)]
    assert np.mean(last) < np.mean(first)


def test_epsilon_decays_and_floors():
    agent = DQNAgent(seed=1)
    start = agent.epsilon
    for _ in range(4000):
        agent.remember(np.zeros(STATE_DIM), 0, 0.0, np.zeros(STATE_DIM), False)
    assert agent.epsilon < start
    assert agent.epsilon >= agent.cfg.epsilon_min


def test_agent_round_trips_through_disk(tmp_path):
    agent = DQNAgent(seed=2)
    for _ in range(100):
        agent.remember(np.random.rand(STATE_DIM), 0, 1.0, np.random.rand(STATE_DIM), False)
    agent.learn()
    path = tmp_path / "agent.json"
    agent.save(path)

    restored = DQNAgent(seed=99)
    assert restored.load(path) is True
    probe = np.random.rand(STATE_DIM)
    assert np.allclose(agent.q_values(probe), restored.q_values(probe))


def test_replay_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=25, seed=1)
    for i in range(120):
        buf.push([i] * STATE_DIM, 0, float(i), [i] * STATE_DIM, False)
    assert len(buf) == 25
    s, a, r, s2, d = buf.sample(10)
    assert s.shape == (10, STATE_DIM) and a.shape == (10,)


# ------------------------------------------------------------- Q-learning

def test_qlearning_updates_its_table():
    agent = QLearningAgent(actions=5, seed=1)
    state = [0.5] * 6
    agent.learn(state, 2, 1.0, state)
    assert agent.table_size >= 1
    assert agent.q_table[agent.discretise(state)][2] > 0


def test_qlearning_discretisation_is_stable():
    agent = QLearningAgent(seed=1)
    a = agent.discretise([0.51, 0.49, 0.72, 0.3, 0.12, 1.0])
    b = agent.discretise([0.51, 0.49, 0.72, 0.3, 0.12, 1.0])
    assert a == b


# ----------------------------------------------------------------- anomaly

def test_anomaly_features_capture_abruptness(frame):
    feats = build_features(frame)
    assert list(feats.columns) == ["cpu_demand", "ram_demand", "cpu_delta",
                                   "cpu_ratio", "ram_ratio"]
    assert not feats.isna().any().any()


def test_detector_flags_an_injected_spike(frame):
    detector = AnomalyDetector("zscore", threshold=4.0)
    detector.train(frame)
    baseline = float(frame["cpu_demand"].median())
    calm = detector.check(baseline, baseline * 2.4,
                          cpu_prev=baseline, cpu_rolling=baseline, ram_rolling=baseline * 2.4)
    spike = detector.check(baseline * 9, baseline * 20,
                           cpu_prev=baseline, cpu_rolling=baseline, ram_rolling=baseline * 2.4)
    assert spike["is_anomaly"] is True
    assert calm["is_anomaly"] is False
    assert spike["severity"] > calm["severity"]


def test_detector_round_trips(tmp_path, frame):
    detector = AnomalyDetector("isolation_forest")
    detector.train(frame)
    detector.save(tmp_path)
    restored = AnomalyDetector.load(tmp_path, "isolation_forest")
    assert restored.method == "isolation_forest"
    assert restored.check(10.0, 20.0)["method"] == "isolation_forest"


# --------------------------------------------------------- trace replay

def test_trace_source_matches_the_generator_contract(tmp_path):
    """Replay must be a drop-in for the generator, or no strategy code is shared."""
    from app.workload import TraceSource

    frame = pd.DataFrame(build_dataset(days=3, seed=5, interval_minutes=5))
    path = tmp_path / "trace.csv"
    frame.to_csv(path, index=False)

    src = TraceSource(path, seed=1, ticks_needed=100)
    gen_keys = set(build_dataset(days=1, seed=1, interval_minutes=5)[0])
    assert gen_keys <= set(src.step()), "trace record is missing generator fields"


def test_trace_seeds_select_different_windows(tmp_path):
    """Identical windows would make a multi-seed study report zero variance."""
    from app.workload import TraceSource

    frame = pd.DataFrame(build_dataset(days=10, seed=5, interval_minutes=5))
    path = tmp_path / "trace.csv"
    frame.to_csv(path, index=False)

    starts = {TraceSource(path, seed=s, ticks_needed=288).start for s in range(6)}
    assert len(starts) > 1


def test_trace_replay_is_reproducible(tmp_path):
    from app.workload import TraceSource

    frame = pd.DataFrame(build_dataset(days=5, seed=5, interval_minutes=5))
    path = tmp_path / "trace.csv"
    frame.to_csv(path, index=False)

    a = [TraceSource(path, seed=3, ticks_needed=50).step() for _ in range(1)]
    b = [TraceSource(path, seed=3, ticks_needed=50).step() for _ in range(1)]
    assert a[0]["cpu_demand"] == b[0]["cpu_demand"]


def test_trace_rejects_a_window_longer_than_the_trace(tmp_path):
    from app.workload import TraceSource

    frame = pd.DataFrame(build_dataset(days=1, seed=5, interval_minutes=5))
    path = tmp_path / "short.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        TraceSource(path, seed=1, ticks_needed=10_000)
