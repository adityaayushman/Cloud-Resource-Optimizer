# Appendix A — Source Listing

Generated from commit `ea61aaa` by `backend/scripts/make_appendix.py`.
Every module below was imported successfully before this document was written, so the listing is executable code rather than a transcription.

7 modules, 1,761 lines. The full repository, including the test suite and the evaluation scripts, is at `github.com/adityaayushman/Cloud-Resource-Optimizer`.

## Contents

A.1  `app/models.py` — Domain types - instance classes, VM and task records

A.2  `app/catalog.py` — Instance catalogue, provider pricing, multi-cloud scoring

A.3  `app/workload.py` — Synthetic workload generator and production-trace replay

A.4  `app/ml/predictor.py` — Demand forecasting, causal features, TreeSHAP, persistence

A.5  `app/ml/dqn.py` — Deep Q-Network in NumPy - manual backprop, Adam, target network

A.6  `app/ml/anomaly.py` — Isolation Forest and z-score detectors, event-based scoring

A.7  `app/ml/forecastability.py` — Pre-training diagnostic: is a forecaster worth building?

---

## A.1  `app/models.py`

*Domain types - instance classes, VM and task records. 150 lines.*

```python
"""Core domain types for the Cloud Resource Optimizer.

Everything the allocator, the RL agent and the API exchange is defined here so
there is a single source of truth for what a VM, a task and a provider are.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CloudProvider(str, Enum):
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"


class InstanceType(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    # Memory-optimised. The general-purpose types all have a RAM:CPU ratio of
    # 2.0, but the workload's ratio is ~2.4, so a fleet built only from them
    # strands roughly 17% of its CPU behind exhausted memory. Mixing in a 4.0
    # ratio node lets the fleet match the workload shape.
    MEMORY = "memory"
    # Compute-dense node. Needed once the workload is a real datacentre trace:
    # the Bitbrains fleet peaks near 300 cores, which the 8-core LARGE cannot
    # serve without a fleet larger than any real operator would run.
    XLARGE = "xlarge"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Region(str, Enum):
    US_EAST = "US-East"
    EU_WEST = "EU-West"
    ASIA_SOUTH = "Asia-South"


_task_counter = itertools.count(1)
_vm_counter = itertools.count(1)


@dataclass
class Task:
    """A unit of work asking for a slice of CPU and RAM for `duration` seconds."""

    cpu_required: float
    ram_required: float
    duration: float = 3600.0
    priority: int = 1
    id: str = field(default_factory=lambda: f"task-{next(_task_counter):06d}")
    status: TaskStatus = TaskStatus.PENDING
    cost: float = 0.0
    assigned_vm_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    # Simulation clock (seconds) at which the task should finish. Set on placement.
    completes_at_tick: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "cpu_required": round(self.cpu_required, 4),
            "ram_required": round(self.ram_required, 4),
            "duration": self.duration,
            "priority": self.priority,
            "status": self.status.value,
            "cost": round(self.cost, 6),
            "assigned_vm_id": self.assigned_vm_id,
        }


@dataclass
class VMInstance:
    """A provisioned virtual machine with finite CPU/RAM capacity."""

    type: InstanceType
    cpu_capacity: float
    ram_capacity: float
    cost_per_hour: float
    provider: CloudProvider = CloudProvider.AWS
    region: Region = Region.US_EAST
    energy_efficiency: float = 1.0
    max_power_watts: float = 100.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    tasks: list[Task] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"vm-{next(_vm_counter):04d}")
    created_at_tick: float = 0.0

    @property
    def cpu_available(self) -> float:
        return max(0.0, self.cpu_capacity - self.cpu_usage)

    @property
    def ram_available(self) -> float:
        return max(0.0, self.ram_capacity - self.ram_usage)

    @property
    def cpu_utilization(self) -> float:
        return (self.cpu_usage / self.cpu_capacity) if self.cpu_capacity else 0.0

    @property
    def ram_utilization(self) -> float:
        return (self.ram_usage / self.ram_capacity) if self.ram_capacity else 0.0

    @property
    def is_idle(self) -> bool:
        return not self.tasks

    def power_watts(self) -> float:
        """Linear power model: 40% idle draw + 60% scaling with CPU load.

        Follows the standard server power approximation used in the
        energy-aware consolidation literature (Beloglazov & Buyya, 2012).
        """
        idle_fraction = 0.40
        load = min(1.0, self.cpu_utilization)
        return (
            self.max_power_watts * idle_fraction
            + self.max_power_watts * (1 - idle_fraction) * load
        ) * self.energy_efficiency

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "provider": self.provider.value,
            "region": self.region.value,
            "cpu_capacity": self.cpu_capacity,
            "ram_capacity": self.ram_capacity,
            "cpu_usage": round(self.cpu_usage, 4),
            "ram_usage": round(self.ram_usage, 4),
            "cpu_utilization": round(self.cpu_utilization * 100, 2),
            "ram_utilization": round(self.ram_utilization * 100, 2),
            "cost_per_hour": round(self.cost_per_hour, 5),
            "power_watts": round(self.power_watts(), 2),
            "task_count": len(self.tasks),
        }
```

---

## A.2  `app/catalog.py`

*Instance catalogue, provider pricing, multi-cloud scoring. 220 lines.*

```python
"""Instance catalogue, provider pricing and the multi-cloud selection layer.

Pricing is *deterministic*: the spot component is derived from a hash of the
provider name and a 5-minute time bucket rather than a fresh random draw, plus a
smooth diurnal term. The same timestamp therefore always yields the same price,
and prices drift continuously rather than jumping - which is what makes the
arbitrage recommendation stable across a page refresh and reproducible in the
evaluation harness. (Prices within one bucket are near-identical but not bitwise
equal, because the diurnal term is continuous in time.)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .models import CloudProvider, InstanceType, Region

# --------------------------------------------------------------------------
# Instance sizing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InstanceSpec:
    cpu: float
    ram: float
    base_cost_per_hour: float
    energy_efficiency: float
    max_power_watts: float


# RAM:CPU ratios - small/medium/large are 2.0, memory is 4.0.
INSTANCE_SPECS: dict[InstanceType, InstanceSpec] = {
    # Smaller instances are more power-efficient per core but cost more per core.
    InstanceType.SMALL: InstanceSpec(2, 4, 0.050, 1.20, 50),
    InstanceType.MEDIUM: InstanceSpec(4, 8, 0.100, 1.00, 100),
    InstanceType.LARGE: InstanceSpec(8, 16, 0.200, 0.80, 200),
    # Memory-optimised: same cores as MEDIUM, double the RAM, ~30% dearer.
    InstanceType.MEMORY: InstanceSpec(4, 16, 0.130, 1.00, 110),
    # Compute-dense: better price and power per core, as real catalogues offer.
    InstanceType.XLARGE: InstanceSpec(16, 32, 0.380, 0.70, 360),
}


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    base_multiplier: float   # baseline on-demand price relative to AWS
    base_latency_ms: float   # median round-trip for the primary region
    volatility: float        # amplitude of the spot-price band
    reliability: float       # modelled availability (fraction)
    pue: float               # power usage effectiveness of the provider's estate


# The three criteria are deliberately independent, each favouring a different
# provider: cost favours GCP, latency favours AWS, carbon favours GCP via PUE.
# An earlier version derived the carbon term from `reliability`, which is not a
# carbon quantity at all - it made the carbon weight a proxy for availability
# and left the carbon slider with no distinguishable effect.
PROVIDER_SPECS: dict[CloudProvider, ProviderSpec] = {
    CloudProvider.AWS:   ProviderSpec(1.00, 38.0, 0.18, 0.9995, 1.14),
    CloudProvider.AZURE: ProviderSpec(0.95, 45.0, 0.10, 0.9990, 1.12),
    CloudProvider.GCP:   ProviderSpec(0.90, 52.0, 0.06, 0.9992, 1.10),
}


@dataclass(frozen=True)
class RegionSpec:
    cost_multiplier: float
    latency_offset_ms: float
    carbon_intensity: float  # kg CO2 per kWh


REGION_SPECS: dict[Region, RegionSpec] = {
    Region.US_EAST:    RegionSpec(1.00, 0.0, 0.40),
    Region.EU_WEST:    RegionSpec(1.05, 12.0, 0.30),
    Region.ASIA_SOUTH: RegionSpec(1.20, 28.0, 0.70),
}

PRICE_BUCKET_SECONDS = 300  # spot prices re-quote every 5 minutes


def _deterministic_unit(seed: str) -> float:
    """Stable pseudo-random value in [0, 1) derived from a string seed."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def price_index(provider: CloudProvider, at: float) -> float:
    """Current price multiplier for a provider at simulation/wall time `at`.

    The result is the provider's baseline multiplier perturbed by a smooth,
    deterministic spot component inside +/- `volatility`.
    """
    spec = PROVIDER_SPECS[provider]
    bucket = int(at // PRICE_BUCKET_SECONDS)
    u = _deterministic_unit(f"{provider.value}:{bucket}")
    # Blend a smooth diurnal term with the bucket noise so the series looks
    # like a real spot market rather than white noise.
    diurnal = math.sin(2 * math.pi * (at % 86400) / 86400.0)
    swing = spec.volatility * (0.6 * (2 * u - 1) + 0.4 * diurnal)
    return round(spec.base_multiplier * (1.0 + swing), 6)


def hourly_cost(
    instance: InstanceType,
    provider: CloudProvider,
    region: Region,
    at: float,
) -> float:
    """On-demand hourly price for one instance."""
    spec = INSTANCE_SPECS[instance]
    return (
        spec.base_cost_per_hour
        * price_index(provider, at)
        * REGION_SPECS[region].cost_multiplier
    )


def latency_ms(provider: CloudProvider, region: Region) -> float:
    return PROVIDER_SPECS[provider].base_latency_ms + REGION_SPECS[region].latency_offset_ms


def carbon_kg_per_kwh(region: Region) -> float:
    """Grid carbon intensity for a region, before provider efficiency."""
    return REGION_SPECS[region].carbon_intensity


def effective_carbon(provider: CloudProvider, region: Region) -> float:
    """kg CO2 per kWh of *useful* compute: grid intensity scaled by PUE.

    PUE is the ratio of total facility power to IT power, so a provider with a
    lower PUE emits less for the same delivered compute in the same region.
    """
    return REGION_SPECS[region].carbon_intensity * PROVIDER_SPECS[provider].pue


# --------------------------------------------------------------------------
# Multi-cloud selection (US-08, US-09, US-18)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectionWeights:
    cost: float = 0.55
    latency: float = 0.25
    carbon: float = 0.20

    def normalised(self) -> "SelectionWeights":
        total = self.cost + self.latency + self.carbon
        if total <= 0:
            return SelectionWeights()
        return SelectionWeights(self.cost / total, self.latency / total, self.carbon / total)


def _min_max(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return lo, (hi if hi > lo else lo + 1e-9)


def score_providers(
    at: float,
    instance: InstanceType = InstanceType.MEDIUM,
    region: Region = Region.US_EAST,
    weights: SelectionWeights | None = None,
) -> list[dict]:
    """Score every provider on cost, latency and carbon.

    Each criterion is min-max normalised across the candidate set so the three
    incommensurable units can be combined; lower score is better.
    """
    w = (weights or SelectionWeights()).normalised()
    candidates = list(PROVIDER_SPECS)

    costs = [hourly_cost(instance, p, region, at) for p in candidates]
    lats = [latency_ms(p, region) for p in candidates]
    carbons = [effective_carbon(p, region) for p in candidates]

    c_lo, c_hi = _min_max(costs)
    l_lo, l_hi = _min_max(lats)
    g_lo, g_hi = _min_max(carbons)

    rows = []
    for provider, cost, lat, carbon in zip(candidates, costs, lats, carbons):
        n_cost = (cost - c_lo) / (c_hi - c_lo)
        n_lat = (lat - l_lo) / (l_hi - l_lo)
        n_carbon = (carbon - g_lo) / (g_hi - g_lo)
        score = w.cost * n_cost + w.latency * n_lat + w.carbon * n_carbon
        rows.append(
            {
                "provider": provider.value,
                "price_index": price_index(provider, at),
                "hourly_cost": round(cost, 5),
                "latency_ms": round(lat, 1),
                "carbon_kg_per_kwh": round(carbon, 4),
                "grid_intensity": round(carbon_kg_per_kwh(region), 3),
                "pue": PROVIDER_SPECS[provider].pue,
                "reliability": PROVIDER_SPECS[provider].reliability,
                "score": round(score, 5),
            }
        )

    rows.sort(key=lambda r: r["score"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def select_provider(
    at: float,
    instance: InstanceType = InstanceType.MEDIUM,
    region: Region = Region.US_EAST,
    weights: SelectionWeights | None = None,
) -> tuple[CloudProvider, list[dict]]:
    """Return the best provider for a new instance plus the full scoreboard."""
    rows = score_providers(at, instance, region, weights)
    return CloudProvider(rows[0]["provider"]), rows
```

---

## A.3  `app/workload.py`

*Synthetic workload generator and production-trace replay. 258 lines.*

```python
"""Synthetic cloud workload generator.

The same generator produces the training dataset and drives the live
simulation, so the model is never evaluated on a distribution it has not seen
and the dashboard is never showing a workload the model was not trained for.

Signal composition (all additive on the task-arrival rate):

  base          constant floor of background traffic
  diurnal       24-hour sinusoid, peaking mid-afternoon
  weekly        weekday/weekend amplitude modulation
  trend         slow linear growth over the observation window
  burst         Poisson-triggered spikes with exponential decay
  noise         Gaussian jitter

CPU and RAM demand are then derived from the arrival rate with per-task
resource intensities that themselves drift slowly, which is what makes the
prediction problem non-trivial.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WorkloadConfig:
    base_tasks: float = 28.0
    diurnal_amplitude: float = 18.0
    weekend_damping: float = 0.55
    trend_per_day: float = 0.35
    # --- non-smooth structure -------------------------------------------
    # Real fleets are not a clean sinusoid. Three conditional effects are
    # modelled because they are what separates a tree ensemble from a linear
    # model: a nightly batch window, a weekday afternoon peak amplification,
    # and a weekly maintenance dip. All three are sharp, hour-conditional
    # steps that a linear model in sin/cos of the hour cannot represent.
    batch_window: tuple[int, int] = (2, 4)      # weekday ETL window [start, end)
    batch_magnitude: float = 24.0
    peak_window: tuple[int, int] = (13, 16)     # weekday afternoon amplification
    peak_amplification: float = 1.28
    maintenance_day: int = 6                    # Sunday
    maintenance_window: tuple[int, int] = (4, 6)
    maintenance_damping: float = 0.40
    saturation_knee: float = 78.0               # soft ceiling on arrivals
    # Tuned so that roughly 2% of intervals carry a genuine spike, matching the
    # anomaly-detector contamination setting. A higher rate makes "anomaly"
    # meaningless - if one interval in seven is a burst, it is the norm.
    burst_probability: float = 0.0032
    burst_magnitude: float = 55.0
    burst_decay: float = 0.62
    burst_label_threshold: float = 8.0
    noise_sigma: float = 2.4
    cpu_per_task_mean: float = 0.42
    ram_per_task_mean: float = 0.85
    intensity_drift: float = 0.06
    interval_minutes: int = 15


class WorkloadGenerator:
    """Stateful generator; call `step()` repeatedly to advance one interval."""

    def __init__(self, config: WorkloadConfig | None = None, seed: int = 42, start_hour: int = 0):
        self.cfg = config or WorkloadConfig()
        self.rng = np.random.default_rng(seed)
        self.t = 0                      # interval index since start
        self.start_hour = start_hour
        self._burst = 0.0
        self._cpu_intensity = self.cfg.cpu_per_task_mean
        self._ram_intensity = self.cfg.ram_per_task_mean

    # -- helpers ---------------------------------------------------------

    @property
    def minutes_elapsed(self) -> float:
        return self.t * self.cfg.interval_minutes

    def _clock(self) -> tuple[float, int, int]:
        total_minutes = self.start_hour * 60 + self.minutes_elapsed
        hour_float = (total_minutes / 60.0) % 24.0
        day_index = int(total_minutes // (60 * 24))
        return hour_float, int(hour_float), day_index % 7

    # -- main step -------------------------------------------------------

    def step(self) -> dict:
        cfg = self.cfg
        hour_float, hour, day_of_week = self._clock()
        day_index = self.minutes_elapsed / (60 * 24)
        is_weekend = day_of_week >= 5

        # Diurnal cycle peaking at ~15:00, trough at ~03:00.
        diurnal = cfg.diurnal_amplitude * math.sin(2 * math.pi * (hour_float - 9.0) / 24.0)
        weekly = cfg.weekend_damping if is_weekend else 1.0
        trend = cfg.trend_per_day * day_index

        # Bursts arrive as a Poisson process and decay geometrically.
        # `onset` marks the interval a burst *starts*, which is the event an
        # anomaly detector can actually catch. The decay tail is elevated but
        # no longer abrupt, so scoring detection against the tail measures the
        # wrong thing.
        onset = False
        if self.rng.random() < cfg.burst_probability:
            if self._burst < 1.0:
                onset = True
            self._burst += cfg.burst_magnitude * (0.5 + self.rng.random())
        self._burst *= cfg.burst_decay
        if self._burst < 0.05:
            self._burst = 0.0

        noise = self.rng.normal(0.0, cfg.noise_sigma)
        num_tasks = (cfg.base_tasks + diurnal + trend) * weekly

        # Nightly batch window (weekdays only): a step, not a curve.
        in_batch = (not is_weekend) and cfg.batch_window[0] <= hour_float < cfg.batch_window[1]
        if in_batch:
            num_tasks += cfg.batch_magnitude

        # Weekday afternoon amplification: an interaction between hour and
        # day type, which a purely additive model cannot express.
        if (not is_weekend) and cfg.peak_window[0] <= hour_float < cfg.peak_window[1]:
            num_tasks *= cfg.peak_amplification

        # Weekly maintenance dip.
        if day_of_week == cfg.maintenance_day and \
                cfg.maintenance_window[0] <= hour_float < cfg.maintenance_window[1]:
            num_tasks *= cfg.maintenance_damping

        num_tasks += self._burst + noise

        # Soft saturation: the ingress tier cannot admit unbounded arrivals.
        knee = cfg.saturation_knee
        if num_tasks > knee:
            num_tasks = knee + (num_tasks - knee) / (1.0 + (num_tasks - knee) / knee)

        num_tasks = float(max(1.0, num_tasks))

        # Per-task resource intensity performs a bounded random walk.
        self._cpu_intensity = float(
            np.clip(
                self._cpu_intensity + self.rng.normal(0, cfg.intensity_drift * 0.1),
                0.18, 0.85,
            )
        )
        self._ram_intensity = float(
            np.clip(
                self._ram_intensity + self.rng.normal(0, cfg.intensity_drift * 0.2),
                0.40, 1.70,
            )
        )

        cpu_demand = num_tasks * self._cpu_intensity + self.rng.normal(0, 0.35)
        ram_demand = num_tasks * self._ram_intensity + self.rng.normal(0, 0.60)

        record = {
            "interval": self.t,
            "minutes": self.minutes_elapsed,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": int(is_weekend),
            "num_tasks": round(num_tasks, 3),
            "cpu_per_task": round(self._cpu_intensity, 4),
            "ram_per_task": round(self._ram_intensity, 4),
            "cpu_demand": round(max(0.1, cpu_demand), 3),
            "ram_demand": round(max(0.2, ram_demand), 3),
            "burst_active": int(self._burst > cfg.burst_label_threshold),
            "burst_onset": int(onset),
        }
        self.t += 1
        return record

    def generate(self, intervals: int) -> list[dict]:
        return [self.step() for _ in range(intervals)]


def build_dataset(days: int = 90, seed: int = 42, interval_minutes: int = 15) -> list[dict]:
    """Produce `days` worth of workload observations."""
    cfg = WorkloadConfig(interval_minutes=interval_minutes)
    gen = WorkloadGenerator(cfg, seed=seed)
    intervals = int(days * 24 * 60 / interval_minutes)
    return gen.generate(intervals)


# ---------------------------------------------------------------------------
# Real-trace replay
# ---------------------------------------------------------------------------

class TraceSource:
    """Replays a recorded workload trace with the same interface as the generator.

    Lets the evaluation harness run identically on synthetic data - where the
    structure is known and controllable - and on a real production trace, where
    it is not. `WorkloadGenerator` and `TraceSource` both expose `step()`
    returning the same record shape, so no strategy code changes.

    **Seeding.** A recorded trace has no randomness, so repeated runs on it
    would be identical and a multi-seed study would report a standard deviation
    of zero, which is meaningless. Each seed therefore starts at a different
    offset in the trace, giving genuinely different demand windows in the same
    way different synthetic seeds give different traces. The offset is drawn
    deterministically from the seed, so runs stay reproducible.
    """

    REQUIRED = ("cpu_demand", "ram_demand", "num_tasks",
                "cpu_per_task", "ram_per_task", "hour", "day_of_week")

    def __init__(self, path, seed: int = 42, ticks_needed: int = 288,
                 random_start: bool = True):
        import pandas as pd

        self.path = str(path)
        df = pd.read_csv(self.path)
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"trace {self.path} is missing columns: {missing}")
        self.frame = df.reset_index(drop=True)

        span = len(self.frame)
        if span < ticks_needed:
            raise ValueError(
                f"trace has {span} rows but {ticks_needed} ticks were requested"
            )

        if random_start and span > ticks_needed:
            rng = np.random.default_rng(seed)
            self.start = int(rng.integers(0, span - ticks_needed))
        else:
            self.start = 0
        self.t = 0

    @property
    def rows(self) -> int:
        return len(self.frame)

    def step(self) -> dict:
        idx = (self.start + self.t) % len(self.frame)
        row = self.frame.iloc[idx]
        self.t += 1
        return {
            "interval": self.t - 1,
            "minutes": (self.t - 1) * 5,
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "is_weekend": int(row.get("is_weekend", int(row["day_of_week"]) >= 5)),
            "num_tasks": float(row["num_tasks"]),
            "cpu_per_task": float(row["cpu_per_task"]),
            "ram_per_task": float(row["ram_per_task"]),
            "cpu_demand": float(row["cpu_demand"]),
            "ram_demand": float(row["ram_demand"]),
            "burst_active": int(row.get("burst_active", 0)),
            "burst_onset": int(row.get("burst_onset", 0)),
        }

    def generate(self, intervals: int) -> list[dict]:
        return [self.step() for _ in range(intervals)]
```

---

## A.4  `app/ml/predictor.py`

*Demand forecasting, causal features, TreeSHAP, persistence. 423 lines.*

```python
"""Workload demand prediction (US-02, US-03, US-10, US-16).

Design notes that differ deliberately from a naive implementation:

* **Causal features only.** Lag and rolling statistics are computed with an
  explicit `.shift(1)` *before* the rolling window, so the feature row for
  time `t` never contains the target value at time `t`.
* **Chronological split.** Train/validation/test are contiguous blocks in time
  order (70/15/15). A shuffled `train_test_split` on a time series lets future
  observations leak into training and inflates the reported score.
* **Tuning on validation, scoring once on test.** Hyperparameters are selected
  by validation MAE; the test block is touched exactly once, at the end.
* **Exact SHAP without the `shap` package.** XGBoost implements TreeSHAP
  natively via `pred_contribs=True`, which keeps the deployed image small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

Algo = Literal["xgboost", "rf", "lr", "persistence"]

FEATURES = [
    "num_tasks",
    "cpu_per_task",
    "ram_per_task",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_weekend",
    "cpu_lag_1",
    "cpu_lag_4",
    "cpu_rolling_mean_4",
    "cpu_rolling_std_8",
    "ram_lag_1",
]

# One-step-ahead forecasting: features observed at interval t predict demand at
# t+1. Predicting demand at t from features at t is curve-fitting, not
# forecasting - and it is useless to an autoscaler, which has to provision
# capacity *before* the demand arrives.
HORIZON = 1
TARGETS = ("target_cpu", "target_ram")
TARGET_LABELS = {"target_cpu": "cpu_demand_t+1", "target_ram": "ram_demand_t+1"}


class PersistenceRegressor:
    """"Next interval equals this one", as a first-class predictor.

    This exists because the cross-dataset study found workloads where persistence
    is the *correct engineering answer* - on Bitbrains no learned model beats it
    at any horizon - and the system had no way to ship that conclusion.
    Persistence was only ever a number in a report, so the recommendation "use
    persistence on this workload" was unactionable.

    **It does not read `cpu_lag_1`.** That column is demand at t-1, and the target
    is demand at t+1, so carrying it forward would be a two-step forecast wearing
    the name of a one-step one - and it scores measurably worse (R2 0.908 against
    the 0.928 the reported baseline achieves). Current demand is not itself a
    feature, but it is recoverable: `num_tasks * cpu_per_task` reconstructs it
    exactly on the production traces (agreement to 1e-13) and to within rounding
    on the synthetic generator, which rounds `num_tasks` to an integer.
    """

    KINDS = {"cpu": ("num_tasks", "cpu_per_task"),
             "ram": ("num_tasks", "ram_per_task")}

    def __init__(self, kind: str = "cpu"):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {sorted(self.KINDS)}, got {kind!r}")
        self.kind = kind
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> "PersistenceRegressor":
        self.feature_names_ = list(X.columns)
        missing = [c for c in self.KINDS[self.kind] if c not in self.feature_names_]
        if missing:
            raise ValueError(
                f"persistence needs {missing} to reconstruct current demand")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        count, per_unit = self.KINDS[self.kind]
        return np.asarray(X[count], dtype=float) * np.asarray(X[per_unit], dtype=float)


@dataclass
class SplitMetrics:
    r2: float
    mae: float
    rmse: float
    mape: float

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> "SplitMetrics":
        err = y_true - y_pred
        denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
        return SplitMetrics(
            r2=float(r2_score(y_true, y_pred)),
            mae=float(mean_absolute_error(y_true, y_pred)),
            rmse=float(np.sqrt(np.mean(err**2))),
            mape=float(np.mean(np.abs(err / denom)) * 100.0),
        )

    def as_dict(self) -> dict:
        return {k: round(v, 5) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal lag/rolling features. Input must be sorted by time ascending."""
    out = df.copy()

    # Cyclical encoding of hour so 23:00 and 00:00 are adjacent.
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    cpu_past = out["cpu_demand"].shift(1)
    out["cpu_lag_1"] = cpu_past
    out["cpu_lag_4"] = out["cpu_demand"].shift(4)
    # shift(1) first => the window ends at t-1 and excludes the target.
    out["cpu_rolling_mean_4"] = cpu_past.rolling(window=4, min_periods=1).mean()
    out["cpu_rolling_std_8"] = cpu_past.rolling(window=8, min_periods=2).std()
    out["ram_lag_1"] = out["ram_demand"].shift(1)

    # Forecasting targets: what demand will be one interval from now.
    out["target_cpu"] = out["cpu_demand"].shift(-HORIZON)
    out["target_ram"] = out["ram_demand"].shift(-HORIZON)

    # Drop the warm-up rows rather than back-filling them: back-filling a lag
    # feature copies a *future* value backwards, which is leakage. The trailing
    # rows go too, because their target does not exist yet.
    required = [c for c in FEATURES if c in out.columns] + list(TARGETS)
    out = out.dropna(subset=required).reset_index(drop=True)
    return out


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class WorkloadPredictor:
    """Predicts next-interval CPU and RAM demand."""

    def __init__(self, algo: Algo = "xgboost"):
        self.algo: Algo = algo
        self.cpu_model = None
        self.ram_model = None
        self.features = list(FEATURES)
        self.metrics: dict = {}
        self.best_params: dict = {}
        self.feature_means: Optional[np.ndarray] = None

    # -- model construction ---------------------------------------------

    def _make_model(self, params: dict | None = None):
        params = params or {}
        if self.algo == "persistence":
            return PersistenceRegressor(**params)
        if self.algo == "xgboost":
            import xgboost as xgb

            defaults = dict(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                # Single-threaded on purpose: the deploy target is a 512 MB
                # shared-CPU container, where extra worker threads cost more
                # memory than they save in wall time on an 8k-row dataset.
                n_jobs=1,
                tree_method="hist",
            )
            defaults.update(params)
            return xgb.XGBRegressor(**defaults)
        if self.algo == "rf":
            defaults = dict(n_estimators=150, max_depth=14, random_state=42, n_jobs=1)
            defaults.update(params)
            return RandomForestRegressor(**defaults)
        return LinearRegression()

    # -- tuning ----------------------------------------------------------

    def _search(self, X_tr, y_tr, X_val, y_val, n_trials: int = 18) -> dict:
        """Random search selected on the *validation* block.

        Boosters are released explicitly between trials; XGBoost holds its
        model state in a native allocation that the Python GC does not free
        promptly, which exhausts a constrained container over many fits.
        """
        if self.algo in ("lr", "persistence"):
            return {}                       # no hyperparameters to search
        import gc

        rng = np.random.default_rng(42)
        best, best_mae = {}, float("inf")
        # Random Forest gets its own budget so the comparison against XGBoost
        # is like-for-like; tuning one arm and not the other would make the
        # ranking an artefact of the search, not of the model.
        if self.algo == "rf":
            n_trials = min(n_trials, 10)
        for _ in range(n_trials):
            if self.algo == "rf":
                params = {
                    "n_estimators": int(rng.integers(120, 300)),
                    "max_depth": int(rng.integers(6, 22)),
                    "min_samples_leaf": int(rng.integers(1, 8)),
                    "max_features": float(rng.uniform(0.4, 1.0)),
                }
            else:
                params = {
                    "n_estimators": int(rng.integers(120, 360)),
                    "max_depth": int(rng.integers(3, 9)),
                    "learning_rate": float(rng.uniform(0.02, 0.22)),
                    "subsample": float(rng.uniform(0.65, 1.0)),
                    "colsample_bytree": float(rng.uniform(0.65, 1.0)),
                    "reg_lambda": float(rng.uniform(0.5, 4.0)),
                }
            model = self._make_model(params)
            try:
                model.fit(X_tr, y_tr)
                mae = mean_absolute_error(y_val, model.predict(X_val))
            except Exception:
                # A single failed trial must not abort the whole search.
                continue
            finally:
                del model
                gc.collect()
            if mae < best_mae:
                best, best_mae = params, mae
        return best

    # -- training --------------------------------------------------------

    def train(self, df: pd.DataFrame, tune: bool = True) -> dict:
        """Fit CPU and RAM models. Returns the metrics report."""
        feat = prepare_features(df)
        train_df, val_df, test_df = chronological_split(feat)

        X_tr, X_val, X_te = (d[self.features] for d in (train_df, val_df, test_df))
        self.feature_means = X_tr.mean().to_numpy(dtype=float)

        report: dict = {"algo": self.algo, "horizon": HORIZON, "n_train": len(train_df),
                        "n_val": len(val_df), "n_test": len(test_df), "targets": {}}

        for target in TARGETS:
            y_tr, y_val, y_te = (d[target] for d in (train_df, val_df, test_df))

            params = self._search(X_tr, y_tr, X_val, y_val) if tune else {}
            if self.algo == "persistence":
                # Which series to carry forward depends on the target, and the
                # estimator cannot infer it from y.
                params = {"kind": "cpu" if target == "target_cpu" else "ram"}
            if target == "target_cpu":
                self.best_params = params

            # Validation score comes from a train-only fit; the deployed model is
            # then refit on train+validation with the chosen hyperparameters.
            val_model = self._make_model(params)
            val_model.fit(X_tr, y_tr)
            val_metrics = SplitMetrics.compute(y_val.to_numpy(), val_model.predict(X_val))

            model = self._make_model(params)
            model.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))

            if target == "target_cpu":
                self.cpu_model = model
            else:
                self.ram_model = model

            # Persistence baseline: "next interval equals this interval". Any
            # model that cannot beat this has learned nothing useful.
            naive = SplitMetrics.compute(y_te.to_numpy(), test_df["cpu_demand"].to_numpy()
                                         if target == "target_cpu"
                                         else test_df["ram_demand"].to_numpy())

            report["targets"][TARGET_LABELS[target]] = {
                "validation": val_metrics.as_dict(),
                "test": SplitMetrics.compute(y_te.to_numpy(), model.predict(X_te)).as_dict(),
                "naive_persistence_test": naive.as_dict(),
            }

        report["best_params"] = self.best_params
        self.metrics = report
        return report

    # -- inference -------------------------------------------------------

    def _frame(self, feature_values: dict) -> pd.DataFrame:
        row = {f: float(feature_values.get(f, 0.0)) for f in self.features}
        return pd.DataFrame([row], columns=self.features)

    def predict(self, feature_values: dict) -> tuple[float, float]:
        if self.cpu_model is None or self.ram_model is None:
            raise RuntimeError("Predictor is not trained. Run scripts/train.py first.")
        X = self._frame(feature_values)
        cpu = float(self.cpu_model.predict(X)[0])
        ram = float(self.ram_model.predict(X)[0])
        return max(0.0, cpu), max(0.0, ram)

    # -- explainability (US-09, US-16) -----------------------------------

    def explain(self, feature_values: dict) -> dict:
        """Per-feature attribution for a single CPU prediction.

        `xgboost` -> exact TreeSHAP (`pred_contribs=True`).
        `lr`      -> exact linear SHAP: phi_i = coef_i * (x_i - E[x_i]).
        `rf`      -> impurity importance fallback, labelled as such.
        """
        X = self._frame(feature_values)

        if self.algo == "xgboost":
            import xgboost as xgb

            booster = self.cpu_model.get_booster()
            contribs = booster.predict(
                xgb.DMatrix(X, feature_names=self.features), pred_contribs=True
            )[0]
            values = contribs[:-1]
            base = float(contribs[-1])
            method = "treeshap-exact"

        elif self.algo == "persistence":
            # The forecast is the product of two features, so the whole prediction
            # is attributable to them and to nothing else. Splitting a product
            # evenly is the Shapley value for two symmetric contributors.
            count, per_unit = PersistenceRegressor.KINDS[self.cpu_model.kind]
            half = float(self.cpu_model.predict(X)[0]) / 2.0
            share = {count: half, per_unit: half}
            values = np.array([share.get(f, 0.0) for f in self.features])
            base = 0.0
            method = "identity-exact"

        elif self.algo == "lr":
            coef = np.asarray(self.cpu_model.coef_, dtype=float)
            means = self.feature_means if self.feature_means is not None else np.zeros(len(coef))
            values = coef * (X.to_numpy(dtype=float)[0] - means)
            base = float(self.cpu_model.intercept_ + float(coef @ means))
            method = "linear-shap-exact"

        else:
            imp = np.asarray(self.cpu_model.feature_importances_, dtype=float)
            pred = float(self.cpu_model.predict(X)[0])
            values = imp * pred
            base = 0.0
            method = "impurity-importance-approx"

        order = np.argsort(-np.abs(values))
        return {
            "method": method,
            "base_value": round(base, 4),
            "contributions": [
                {
                    "feature": self.features[i],
                    "value": round(float(X.iloc[0, i]), 4),
                    "contribution": round(float(values[i]), 5),
                }
                for i in order
            ],
        }

    # -- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.cpu_model, directory / f"cpu_{self.algo}.joblib")
        joblib.dump(self.ram_model, directory / f"ram_{self.algo}.joblib")
        payload = {
            "algo": self.algo,
            "features": self.features,
            "metrics": self.metrics,
            "best_params": self.best_params,
            "feature_means": None if self.feature_means is None else self.feature_means.tolist(),
        }
        (directory / f"predictor_{self.algo}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, algo: Algo = "xgboost") -> "WorkloadPredictor":
        import joblib

        meta_path = directory / f"predictor_{algo}.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No trained {algo} predictor in {directory}. Run scripts/train.py."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obj = cls(algo=algo)
        obj.features = meta["features"]
        obj.metrics = meta.get("metrics", {})
        obj.best_params = meta.get("best_params", {})
        means = meta.get("feature_means")
        obj.feature_means = None if means is None else np.asarray(means, dtype=float)
        obj.cpu_model = joblib.load(directory / f"cpu_{algo}.joblib")
        obj.ram_model = joblib.load(directory / f"ram_{algo}.joblib")
        return obj
```

---

## A.5  `app/ml/dqn.py`

*Deep Q-Network in NumPy - manual backprop, Adam, target network. 310 lines.*

```python
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
```

---

## A.6  `app/ml/anomaly.py`

*Isolation Forest and z-score detectors, event-based scoring. 197 lines.*

```python
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
```

---

## A.7  `app/ml/forecastability.py`

*Pre-training diagnostic: is a forecaster worth building?. 203 lines.*

```python
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

# Which predictor the verdict implies, and why.
#
# The choice of `lr` rather than `xgboost` is the uncomfortable one, and it is
# what the measurement says: across the four production traces, all seven
# significant wins over persistence belong to linear regression, and neither tree
# ensemble beat persistence on a real trace at *any* horizon. On Bitbrains the
# ensembles lose by more than the linear model does. Mean reversion is linear
# structure; trees split on local thresholds and extrapolate noise.
#
# `xgboost` remains the shipped default because it is what the system was built
# and tuned around, and because it does win on the synthetic workload, whose
# generator contains learnable step structure. This function is the honest
# recommendation, not a silent override - see `docs/RESULTS-CROSS-DATASET.md`.
RECOMMENDED_ALGO = {
    "model_likely_helps": "lr",
    "persistence_sufficient": "persistence",
    "inconclusive": None,
}


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _inconclusive(reason: str, samples: int) -> dict:
    """Every return path carries the same keys, so callers need no special case."""
    return {
        "verdict": "inconclusive",
        "reason": reason,
        "recommended_algo": RECOMMENDED_ALGO["inconclusive"],
        "recommendation_note": _recommendation_note("inconclusive"),
        "samples": int(samples),
    }


def _recommendation_note(verdict: str) -> str:
    if verdict == "model_likely_helps":
        return (
            "Linear regression, not a tree ensemble. Across four production "
            "traces every significant win over persistence belonged to linear "
            "regression, and neither XGBoost nor Random Forest beat persistence "
            "on a real trace at any horizon. Mean reversion is linear structure."
        )
    if verdict == "persistence_sufficient":
        return (
            "Ship the persistence predictor (algo='persistence'). It needs no "
            "training, no artifact and no retraining schedule, and on a workload "
            "shaped like this one nothing measured has beaten it."
        )
    return (
        "No recommendation. Train one model and compare its MAE against the "
        "persistence baseline on held-out blocks before committing to a pipeline."
    )


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
        return _inconclusive(
            f"only {len(series)} samples; at least {MIN_SAMPLES} "
            f"({MIN_SAMPLES * interval_minutes / 60:.0f} h) are needed before this "
            f"statistic is stable", len(series))

    mean = float(series.mean())
    if mean <= 0:
        return _inconclusive("demand is zero or negative", len(series))

    diff = np.diff(series)
    diff_acf1 = _autocorr(diff, 1)
    if not np.isfinite(diff_acf1):
        return _inconclusive("demand is constant", len(series))

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
        "recommended_algo": RECOMMENDED_ALGO[verdict],
        "recommendation_note": _recommendation_note(verdict),
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
```
