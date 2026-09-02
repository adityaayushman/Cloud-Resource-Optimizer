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
