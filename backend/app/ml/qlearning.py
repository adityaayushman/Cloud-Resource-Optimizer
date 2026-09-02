"""Tabular Q-learning agent (blueprint 6.4).

Kept as a genuine comparison arm for the ablation study rather than dead code:
`ablation.py` runs a `q_learning` configuration against the DQN configuration so
the report can state what the deep network actually buys over a discretised
Q-table on the same environment.

State is discretised into coarse buckets so it fits a dictionary; that
discretisation is exactly why it plateaus below the DQN on a continuous
state space, which is the point the comparison makes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.999,
        seed: int = 42,
    ):
        self.q_table: dict[tuple, np.ndarray] = {}
        self.actions = list(range(actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def discretise(self, state: list[float]) -> tuple:
        """Bucket the continuous observation so it can key a Q-table.

        The DQN consumes `state` directly; this agent must round it, which is
        the structural difference between the two.
        """
        return (
            round(float(state[0]) * 4),   # predicted CPU / capacity
            round(float(state[1]) * 4),   # predicted RAM / capacity
            round(float(state[2]) * 5),   # CPU utilisation
            round(float(state[4]) * 8),   # fleet size
        )

    def _row(self, key: tuple) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(self.actions))
        return self.q_table[key]

    def act(self, state: list[float], greedy: bool = False) -> int:
        key = self.discretise(state)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self._row(key)))

    def learn(self, state, action: int, reward: float, next_state, done: bool = False) -> float:
        key, next_key = self.discretise(state), self.discretise(next_state)
        row, next_row = self._row(key), self._row(next_key)
        predicted = row[action]
        target = reward + (0.0 if done else self.gamma * float(np.max(next_row)))
        row[action] += self.alpha * (target - predicted)
        self.updates += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(abs(target - predicted))

    @property
    def table_size(self) -> int:
        return len(self.q_table)

    def snapshot(self) -> dict:
        """Deep copy of the table, for held-out checkpoint selection."""
        return {"q_table": {k: v.copy() for k, v in self.q_table.items()},
                "epsilon": self.epsilon}

    def restore(self, snap: dict) -> None:
        self.q_table = {k: v.copy() for k, v in snap["q_table"].items()}
        self.epsilon = snap["epsilon"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "actions": len(self.actions),
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
            "q_table": {"|".join(map(str, k)): v.tolist() for k, v in self.q_table.items()},
        }), encoding="utf-8")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self.epsilon = float(data.get("epsilon", self.epsilon_min))
        self.q_table = {
            tuple(int(p) for p in k.split("|")): np.asarray(v, dtype=float)
            for k, v in data.get("q_table", {}).items()
        }
        return True
