"""Aggregated ML layer - the `ml_models` module from the report's module map.

The implementation is split across `app/ml/` for readability; this module is the
single import surface the engine and the API use, matching the blueprint's
four-module architecture (models / ml_models / engine / app).

    from .ml_models import WorkloadPredictor, DQNAgent, AnomalyDetector, QLearningAgent
"""

from .ml.anomaly import AnomalyDetector
from .ml.dqn import (
    ACTION_DIM,
    ACTION_NAMES,
    HEADROOM_LEVELS,
    STATE_DIM,
    DQNAgent,
    DQNConfig,
    MLP,
    ReplayBuffer,
)
from .ml.predictor import (
    FEATURES,
    TARGETS,
    SplitMetrics,
    WorkloadPredictor,
    chronological_split,
    prepare_features,
)
from .ml.qlearning import QLearningAgent

__all__ = [
    "WorkloadPredictor", "FEATURES", "TARGETS", "SplitMetrics",
    "prepare_features", "chronological_split",
    "DQNAgent", "DQNConfig", "MLP", "ReplayBuffer",
    "STATE_DIM", "ACTION_DIM", "ACTION_NAMES", "HEADROOM_LEVELS",
    "AnomalyDetector", "QLearningAgent",
]
