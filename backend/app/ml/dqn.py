"""Deep Q-Network agent for adaptive fleet sizing (US-11, US-06/US-07 Sprint II).

Implemented directly in NumPy rather than PyTorch. Two reasons:

1. **Deployment footprint.** The CPU-only torch wheel is ~800 MB installed and
   ~250 MB resident on import, which does not fit a 512 MB free-tier container.
   The whole backend including xgboost and scikit-learn now installs in ~180 MB.
2. **Transparency.** The forward pass, the Bellman target and the gradient are
   all visible in one file, which is easier to defend than a framework call.

The algorithm is standard DQN (Mnih et al., 2015) with the pieces that are
usually got wrong made explicit:

* the temporal-difference target is computed from a **separate target network**
  that is hard-synced every `target_sync` learning steps;
* the target is a plain array, so no gradient can flow into it;
* the loss is applied **only to the Q-value of the action actually taken**;
* transitions are sampled as a **batch**, not looped one at a time;
* terminal transitions drop the bootstrap term via the `done` flag.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Action semantics: each action is a *headroom setpoint*. The agent chooses how
# much capacity to hold relative to forecast demand, and the fleet is resized to
# match in a single step.
#
# The obvious alternative - incremental actions (add one node / remove one node)
# - was tried first and fails badly. From an undersized fleet the agent needs
# several consecutive "add" actions to reach a good configuration, and every
# intermediate step is still undersized and still scores badly, so there is no
# gradient to follow. With epsilon at 0.05 the chance of randomly stringing four
# correct actions together is ~1e-6, and the policy collapses into a one-node
# cluster that drops most of the workload. Setpoint actions remove the credit
# assignment problem: one action, one complete, immediately-scored decision.
# The lowest setpoint is 1.00, not below it. A setpoint under 1.0 provisions
# less capacity than the forecast already calls for, which guarantees rejected
# work before forecast error or packing loss is even considered - there is no
# state in which it is the right answer, and leaving it available let the agent
# buy a cheap fleet at the cost of a 13% task failure rate.
HEADROOM_LEVELS = [1.00, 1.15, 1.30, 1.50, 1.75]
ACTION_NAMES = [f"headroom_{h:.2f}x" for h in HEADROOM_LEVELS]

STATE_DIM = 6
ACTION_DIM = len(HEADROOM_LEVELS)


# ---------------------------------------------------------------------------
# Multilayer perceptron with manual backprop + Adam
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, sizes: list[int], seed: int = 42, lr: float = 1e-3):
        self.sizes = sizes
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation, appropriate for ReLU.
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """Returns (output, cache) where cache holds pre/post activations."""
        cache = []
        a = X
        n_layers = len(self.W)
        for i in range(n_layers):
            z = a @ self.W[i] + self.b[i]
            cache.append((a, z))
            a = np.maximum(z, 0.0) if i < n_layers - 1 else z  # linear head
        return a, cache

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def backward(self, cache: list, d_out: np.ndarray) -> tuple[list, list]:
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = d_out
        for i in reversed(range(len(self.W))):
            a_prev, z = cache[i]
            if i < len(self.W) - 1:
                delta = delta * (z > 0)
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T
        return grads_W, grads_b

    def adam_step(self, grads_W, grads_b, clip: float = 10.0):
        b1, b2, eps = 0.9, 0.999, 1e-8
        self._t += 1
        for i in range(len(self.W)):
            gW = np.clip(grads_W[i], -clip, clip)
            gb = np.clip(grads_b[i], -clip, clip)

            self._mW[i] = b1 * self._mW[i] + (1 - b1) * gW
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * (gW**2)
            mW_hat = self._mW[i] / (1 - b1**self._t)
            vW_hat = self._vW[i] / (1 - b2**self._t)
            self.W[i] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self._mb[i] = b1 * self._mb[i] + (1 - b1) * gb
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * (gb**2)
            mb_hat = self._mb[i] / (1 - b1**self._t)
            vb_hat = self._vb[i] / (1 - b2**self._t)
            self.b[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def copy_from(self, other: "MLP") -> None:
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def state(self) -> dict:
        return {
            "sizes": self.sizes,
            "W": [w.tolist() for w in self.W],
            "b": [b.tolist() for b in self.b],
        }

    @classmethod
    def from_state(cls, state: dict, lr: float = 1e-3) -> "MLP":
        net = cls(state["sizes"], lr=lr)
        net.W = [np.asarray(w, dtype=float) for w in state["W"]]
        net.b = [np.asarray(b, dtype=float) for b in state["b"]]
        return net


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 20000, seed: int = 42):
        self.buf: deque = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        self.buf.append((np.asarray(s, dtype=float), int(a), float(r),
                         np.asarray(s2, dtype=float), bool(done)))

    def sample(self, batch_size: int):
        idx = self.rng.choice(len(self.buf), size=batch_size, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([b[0] for b in items])
        a = np.array([b[1] for b in items], dtype=int)
        r = np.array([b[2] for b in items], dtype=float)
        s2 = np.stack([b[3] for b in items])
        d = np.array([b[4] for b in items], dtype=float)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.997     # applied once per environment step
    batch_size: int = 64
    warmup: int = 256                # transitions before learning starts
    target_sync: int = 200           # learning steps between hard target syncs
    hidden: tuple[int, int] = (64, 64)
    huber_kappa: float = 1.0


class DQNAgent:
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        config: DQNConfig | None = None,
        seed: int = 42,
    ):
        self.cfg = config or DQNConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        sizes = [state_dim, *self.cfg.hidden, action_dim]
        self.online = MLP(sizes, seed=seed, lr=self.cfg.lr)
        self.target = MLP(sizes, seed=seed + 1, lr=self.cfg.lr)
        self.target.copy_from(self.online)          # start synced
        self.memory = ReplayBuffer(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.epsilon = self.cfg.epsilon_start
        self.learn_steps = 0
        self.env_steps = 0
        self.loss_history: list[float] = []

    # -- acting ----------------------------------------------------------

    def q_values(self, state) -> np.ndarray:
        return self.online(np.asarray(state, dtype=float).reshape(1, -1))[0]

    def act(self, state, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        return int(np.argmax(self.q_values(state)))

    def remember(self, s, a, r, s2, done) -> None:
        self.memory.push(s, a, r, s2, done)
        self.env_steps += 1
        if self.epsilon > self.cfg.epsilon_min:
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    # -- learning --------------------------------------------------------

    def learn(self) -> float | None:
        cfg = self.cfg
        if len(self.memory) < max(cfg.warmup, cfg.batch_size):
            return None

        s, a, r, s2, done = self.memory.sample(cfg.batch_size)

        # Bellman target from the *target* network. Plain ndarray -> no
        # gradient path back into the target, which is the point of DQN.
        next_q = self.target(s2)
        target_q = r + cfg.gamma * np.max(next_q, axis=1) * (1.0 - done)

        pred_all, cache = self.online.forward(s)
        rows = np.arange(cfg.batch_size)
        pred = pred_all[rows, a]

        # Huber gradient wrt the predicted Q of the taken action only.
        diff = pred - target_q
        grad = np.where(np.abs(diff) <= cfg.huber_kappa, diff,
                        cfg.huber_kappa * np.sign(diff))

        d_out = np.zeros_like(pred_all)
        d_out[rows, a] = grad / cfg.batch_size

        gW, gb = self.online.backward(cache, d_out)
        self.online.adam_step(gW, gb)

        self.learn_steps += 1
        if self.learn_steps % cfg.target_sync == 0:
            self.target.copy_from(self.online)      # hard sync

        loss = float(np.mean(np.where(
            np.abs(diff) <= cfg.huber_kappa,
            0.5 * diff**2,
            cfg.huber_kappa * (np.abs(diff) - 0.5 * cfg.huber_kappa),
        )))
        self.loss_history.append(loss)
        return loss

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "online": self.online.state(),
            "target": self.target.state(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "env_steps": self.env_steps,
            "config": {
                "lr": self.cfg.lr, "gamma": self.cfg.gamma,
                "epsilon_min": self.cfg.epsilon_min,
                "batch_size": self.cfg.batch_size,
                "target_sync": self.cfg.target_sync,
            },
        }), encoding="utf-8")

    def snapshot(self) -> dict:
        """Deep copy of the learnable parameters, for checkpoint selection."""
        return {
            "online": {"W": [w.copy() for w in self.online.W],
                       "b": [b.copy() for b in self.online.b]},
            "target": {"W": [w.copy() for w in self.target.W],
                       "b": [b.copy() for b in self.target.b]},
            "epsilon": self.epsilon,
        }

    def restore(self, snap: dict) -> None:
        self.online.W = [w.copy() for w in snap["online"]["W"]]
        self.online.b = [b.copy() for b in snap["online"]["b"]]
        self.target.W = [w.copy() for w in snap["target"]["W"]]
        self.target.b = [b.copy() for b in snap["target"]["b"]]
        self.epsilon = snap["epsilon"]

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self.online = MLP.from_state(data["online"], lr=self.cfg.lr)
        self.target = MLP.from_state(data["target"], lr=self.cfg.lr)
        self.epsilon = float(data.get("epsilon", self.cfg.epsilon_min))
        self.learn_steps = int(data.get("learn_steps", 0))
        self.env_steps = int(data.get("env_steps", 0))
        return True
