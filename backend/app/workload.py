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
