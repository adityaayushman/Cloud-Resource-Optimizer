"""Workload anomaly detection (US-14, US-15).

**Feature choice is the whole problem here.** Running a detector on raw
`(cpu_demand, ram_demand)` does not work: a burst at 09:00 has the same
magnitude as a perfectly normal 15:00 peak, so no density-based method can
separate them, and measured F1 lands near 0.10. What makes a burst anomalous is
that it is *sudden* and *out of line with its own recent history*, so the
detector is given:

    cpu_demand      level
    ram_demand      level
    cpu_delta       first difference (how abruptly it moved)
    cpu_ratio       level / trailing mean (how far above its own baseline)
    ram_ratio       same for memory

Two interchangeable detectors sit on top of those features:

* ``isolation_forest`` - unsupervised, multivariate.
* ``zscore``           - flags a value beyond `threshold` sigma on any feature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

Method = Literal["isolation_forest", "zscore"]

FEATURES = ["cpu_demand", "ram_demand", "cpu_delta", "cpu_ratio", "ram_ratio"]
ROLLING_WINDOW = 6


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the detector's feature frame from a raw workload history."""
    out = pd.DataFrame(index=df.index)
    cpu = df["cpu_demand"].astype(float)
    ram = df["ram_demand"].astype(float)

    # Trailing statistics exclude the current point (shift(1)) so the ratio
    # measures departure from the past, not from a window containing itself.
    cpu_roll = cpu.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    ram_roll = ram.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()

    out["cpu_demand"] = cpu
    out["ram_demand"] = ram
    out["cpu_delta"] = cpu.diff().fillna(0.0)
    out["cpu_ratio"] = (cpu / cpu_roll.replace(0, np.nan)).fillna(1.0)
    out["ram_ratio"] = (ram / ram_roll.replace(0, np.nan)).fillna(1.0)
    return out[FEATURES]


def context_row(
    cpu_demand: float,
    ram_demand: float,
    cpu_prev: Optional[float] = None,
    cpu_rolling: Optional[float] = None,
    ram_rolling: Optional[float] = None,
) -> np.ndarray:
    """Build a single feature row from live values plus recent history."""
    cpu_prev = cpu_demand if cpu_prev is None else cpu_prev
    cpu_rolling = cpu_demand if not cpu_rolling else cpu_rolling
    ram_rolling = ram_demand if not ram_rolling else ram_rolling
    return np.array([[
        float(cpu_demand),
        float(ram_demand),
        float(cpu_demand - cpu_prev),
        float(cpu_demand / cpu_rolling) if cpu_rolling else 1.0,
        float(ram_demand / ram_rolling) if ram_rolling else 1.0,
    ]])


class AnomalyDetector:
    # Operating points chosen from the sweep in docs/RESULTS.md. Both are set
    # to the same event recall (~0.85) so the two methods can be compared
    # like-for-like; recall is favoured over precision because a missed demand
    # surge costs an SLA breach while a false alarm costs one operator glance.
    def __init__(self, method: Method = "isolation_forest", contamination: float = 0.008,
                 threshold: float = 4.0):
        self.method: Method = method
        self.contamination = contamination
        self.threshold = threshold
        self.model: IsolationForest | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def train(self, df: pd.DataFrame) -> dict:
        feats = build_features(df)
        X = feats.to_numpy(dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination, random_state=42,
                n_estimators=150, n_jobs=1,
            )
            self.model.fit(X)
            flagged = int((self.model.predict(X) == -1).sum())
        else:
            std = np.where(self.std == 0, 1e-9, self.std)
            z = np.abs((X - self.mean) / std)
            flagged = int((z > self.threshold).any(axis=1).sum())

        return {
            "method": self.method,
            "features": FEATURES,
            "n_train": len(df),
            "flagged_in_training": flagged,
            "flagged_rate": round(flagged / max(1, len(df)), 4),
        }

    # -- inference -------------------------------------------------------

    def _score_row(self, X: np.ndarray) -> dict:
        if self.method == "isolation_forest":
            if self.model is None:
                raise RuntimeError("Anomaly detector is not trained.")
            is_anom = bool(self.model.predict(X)[0] == -1)
            raw = float(self.model.score_samples(X)[0])
            severity = float(np.clip((-raw - 0.40) / 0.30, 0.0, 1.0))
            detail = {"anomaly_score": round(raw, 5)}
        else:
            if self.mean is None or self.std is None:
                raise RuntimeError("Anomaly detector is not trained.")
            std = np.where(self.std == 0, 1e-9, self.std)
            z = np.abs((X - self.mean) / std)[0]
            is_anom = bool((z > self.threshold).any())
            severity = float(np.clip(z.max() / (2 * self.threshold), 0.0, 1.0))
            detail = {
                "max_z": round(float(z.max()), 3),
                "z_by_feature": {f: round(float(v), 3) for f, v in zip(FEATURES, z)},
            }
        return {
            "is_anomaly": is_anom,
            "method": self.method,
            "severity": round(severity, 4),
            **detail,
        }

    def check(
        self,
        cpu_demand: float,
        ram_demand: float,
        cpu_prev: Optional[float] = None,
        cpu_rolling: Optional[float] = None,
        ram_rolling: Optional[float] = None,
    ) -> dict:
        return self._score_row(
            context_row(cpu_demand, ram_demand, cpu_prev, cpu_rolling, ram_rolling)
        )

    def check_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised evaluation over a whole history - used for scoring."""
        X = build_features(df).to_numpy(dtype=float)
        if self.method == "isolation_forest":
            if self.model is None:
                raise RuntimeError("Anomaly detector is not trained.")
            return self.model.predict(X) == -1
        std = np.where(self.std == 0, 1e-9, self.std)
        return (np.abs((X - self.mean) / std) > self.threshold).any(axis=1)

    # -- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"anomaly_{self.method}.json").write_text(json.dumps({
            "method": self.method,
            "features": FEATURES,
            "contamination": self.contamination,
            "threshold": self.threshold,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
        }, indent=2), encoding="utf-8")
        if self.method == "isolation_forest":
            joblib.dump(self.model, directory / "anomaly_isolation_forest.joblib")

    @classmethod
    def load(cls, directory: Path, method: Method = "isolation_forest") -> "AnomalyDetector":
        import joblib

        meta_path = directory / f"anomaly_{method}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No trained {method} detector in {directory}.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obj = cls(method=method, contamination=meta["contamination"], threshold=meta["threshold"])
        obj.mean = None if meta["mean"] is None else np.asarray(meta["mean"], dtype=float)
        obj.std = None if meta["std"] is None else np.asarray(meta["std"], dtype=float)
        if method == "isolation_forest":
            obj.model = joblib.load(directory / "anomaly_isolation_forest.joblib")
        return obj
