"""Resource allocation, autoscaling and advisory logic.

`ResourceAllocator` owns the fleet and the placement policy.
`SmartAllocator` adds the ML predictor, the anomaly detector and the DQN agent.
`AutoScaler` provides the reactive/predictive baselines used in the ablation.
`AdvisoryEngine` turns the current + predicted state into operator guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from .catalog import (
    INSTANCE_SPECS,
    SelectionWeights,
    carbon_kg_per_kwh,
    hourly_cost,
    latency_ms,
    score_providers,
    select_provider,
)
from .models import CloudProvider, InstanceType, Region, Task, TaskStatus, VMInstance
from .ml.dqn import ACTION_NAMES, HEADROOM_LEVELS

Strategy = Literal["first_fit", "best_fit", "worst_fit", "cost_aware"]

MAX_FLEET = 40          # hard cap: an unbounded scale-up loop is a billing incident
MIN_FLEET = 1
TARGET_UTILISATION = 0.72  # achievable given packing fragmentation; see pick_instance_type
SLA_CRITICAL_UTILISATION = 0.92


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------

class ResourceAllocator:
    def __init__(self, region: Region = Region.US_EAST, multi_cloud: bool = True):
        self.vms: list[VMInstance] = []
        self.region = region
        self.multi_cloud = multi_cloud
        self.clock: float = 0.0            # simulation seconds
        self.accrued_cost: float = 0.0     # dollars spent so far
        self.accrued_energy_kwh: float = 0.0
        self.accrued_co2_kg: float = 0.0
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0
        self.placed_tasks: int = 0
        self.sla_breach_ticks: int = 0
        self.total_ticks: int = 0

    # -- fleet mutation --------------------------------------------------

    def add_vm(
        self,
        vm_type: InstanceType,
        provider: Optional[CloudProvider] = None,
        region: Optional[Region] = None,
        weights: SelectionWeights | None = None,
    ) -> Optional[VMInstance]:
        """Provision one VM. Provider is chosen by the multi-cloud layer unless pinned."""
        if len(self.vms) >= MAX_FLEET:
            return None

        region = region or self.region
        if provider is None:
            if self.multi_cloud:
                provider, _ = select_provider(self.clock, vm_type, region, weights)
            else:
                provider = CloudProvider.AWS

        spec = INSTANCE_SPECS[vm_type]
        vm = VMInstance(
            type=vm_type,
            cpu_capacity=spec.cpu,
            ram_capacity=spec.ram,
            cost_per_hour=hourly_cost(vm_type, provider, region, self.clock),
            provider=provider,
            region=region,
            energy_efficiency=spec.energy_efficiency,
            max_power_watts=spec.max_power_watts,
            created_at_tick=self.clock,
        )
        self.vms.append(vm)
        return vm

    def remove_vm(self, vm_id: str, reallocate: bool = True) -> bool:
        """Decommission a VM, re-placing its running tasks onto the rest of the fleet."""
        for i, vm in enumerate(self.vms):
            if vm.id != vm_id:
                continue
            if len(self.vms) <= MIN_FLEET:
                return False
            displaced = list(vm.tasks)
            del self.vms[i]
            for task in displaced:
                task.assigned_vm_id = None
                task.status = TaskStatus.PENDING
                if reallocate:
                    self.allocate_task(task, count_as_new=False)
                else:
                    task.status = TaskStatus.FAILED
                    self.failed_tasks += 1
            return True
        return False

    def remove_least_utilised(self) -> Optional[str]:
        """Pick the cheapest VM to retire: idle first, then lowest utilisation."""
        if len(self.vms) <= MIN_FLEET:
            return None
        idle = [vm for vm in self.vms if vm.is_idle]
        target = min(idle or self.vms, key=lambda v: (v.cpu_utilization, -v.cost_per_hour))
        return target.id if self.remove_vm(target.id) else None

    # -- placement -------------------------------------------------------

    def _fits(self, vm: VMInstance, task: Task) -> bool:
        return vm.cpu_available >= task.cpu_required and vm.ram_available >= task.ram_required

    def _pick_vm(self, task: Task, strategy: Strategy) -> Optional[VMInstance]:
        candidates = [vm for vm in self.vms if self._fits(vm, task)]
        if not candidates:
            return None
        if strategy == "first_fit":
            return candidates[0]
        if strategy == "worst_fit":
            return max(candidates, key=lambda v: v.cpu_available)
        if strategy == "cost_aware":
            # Cheapest marginal cost for this task, tie-broken by tight packing.
            return min(
                candidates,
                key=lambda v: (self._task_cost(v, task), v.cpu_available),
            )
        # best_fit: leave the least slack behind
        return min(
            candidates,
            key=lambda v: (v.cpu_available - task.cpu_required)
            + (v.ram_available - task.ram_required),
        )

    def _task_cost(self, vm: VMInstance, task: Task) -> float:
        """Charge the task its proportional share of the VM for its duration."""
        cpu_share = task.cpu_required / vm.cpu_capacity
        ram_share = task.ram_required / vm.ram_capacity
        return ((cpu_share + ram_share) / 2.0) * vm.cost_per_hour * (task.duration / 3600.0)

    def allocate_task(
        self, task: Task, strategy: Strategy = "best_fit", count_as_new: bool = True
    ) -> bool:
        vm = self._pick_vm(task, strategy)
        if vm is None:
            task.status = TaskStatus.FAILED
            if count_as_new:
                self.failed_tasks += 1
            return False

        vm.cpu_usage += task.cpu_required
        vm.ram_usage += task.ram_required
        vm.tasks.append(task)
        task.assigned_vm_id = vm.id
        task.status = TaskStatus.RUNNING
        task.cost = self._task_cost(vm, task)
        task.completes_at_tick = self.clock + task.duration
        if count_as_new:
            self.placed_tasks += 1
        return True

    def simulate_fault(self, vm_id: str) -> dict:
        """Kill a VM and report how many of its tasks survived re-placement."""
        vm = next((v for v in self.vms if v.id == vm_id), None)
        if vm is None:
            return {"ok": False, "reason": "vm_not_found"}
        displaced = len(vm.tasks)
        before_failed = self.failed_tasks
        self.remove_vm(vm_id, reallocate=True)
        lost = self.failed_tasks - before_failed
        return {
            "ok": True,
            "vm_id": vm_id,
            "tasks_displaced": displaced,
            "tasks_recovered": displaced - lost,
            "tasks_lost": lost,
        }

    # -- clock -----------------------------------------------------------

    def retire_completed(self) -> int:
        """Release tasks whose duration has elapsed. Returns how many finished."""
        completed = 0
        for vm in self.vms:
            still_running = []
            for task in vm.tasks:
                if task.completes_at_tick is not None and task.completes_at_tick <= self.clock:
                    vm.cpu_usage = max(0.0, vm.cpu_usage - task.cpu_required)
                    vm.ram_usage = max(0.0, vm.ram_usage - task.ram_required)
                    task.status = TaskStatus.COMPLETED
                    completed += 1
                else:
                    still_running.append(task)
            vm.tasks = still_running
        self.completed_tasks += completed
        return completed

    def tick(self, seconds: float = 900.0, retire: bool = True) -> dict:
        """Advance the simulation: accrue cost/energy and optionally retire work.

        `retire=False` advances the clock and bills the interval but leaves the
        current cohort resident. The API uses that so a caller inspecting the
        fleet between requests sees the work that was just placed rather than an
        idle cluster - utilisation of a fleet sampled after its workload has
        been released is always zero, which is true and useless.
        """
        hours = seconds / 3600.0
        hourly = sum(vm.cost_per_hour for vm in self.vms)
        self.accrued_cost += hourly * hours

        watts = sum(vm.power_watts() for vm in self.vms)
        kwh = watts / 1000.0 * hours
        self.accrued_energy_kwh += kwh
        self.accrued_co2_kg += kwh * carbon_kg_per_kwh(self.region)

        self.clock += seconds
        self.total_ticks += 1

        completed = self.retire_completed() if retire else 0

        if self._sla_violations() > 0:
            self.sla_breach_ticks += 1

        return {"completed": completed, "clock": self.clock,
                "hourly_cost": round(hourly, 5)}

    # -- metrics ---------------------------------------------------------

    def _sla_violations(self) -> int:
        return sum(1 for vm in self.vms if vm.cpu_utilization > SLA_CRITICAL_UTILISATION)

    def capacity(self) -> tuple[float, float]:
        return (
            sum(vm.cpu_capacity for vm in self.vms),
            sum(vm.ram_capacity for vm in self.vms),
        )

    def usage(self) -> tuple[float, float]:
        return (
            sum(vm.cpu_usage for vm in self.vms),
            sum(vm.ram_usage for vm in self.vms),
        )

    def get_metrics(self) -> dict:
        cap_cpu, cap_ram = self.capacity()
        use_cpu, use_ram = self.usage()

        cpu_util = (use_cpu / cap_cpu * 100.0) if cap_cpu else 0.0
        ram_util = (use_ram / cap_ram * 100.0) if cap_ram else 0.0
        wastage = 100.0 - (cpu_util + ram_util) / 2.0 if cap_cpu and cap_ram else 0.0

        hourly = sum(vm.cost_per_hour for vm in self.vms)
        watts = sum(vm.power_watts() for vm in self.vms)
        co2_hr = watts / 1000.0 * carbon_kg_per_kwh(self.region)

        violations = self._sla_violations()
        sla = 100.0 - (violations / len(self.vms) * 100.0) if self.vms else 100.0
        pressure = max((vm.cpu_utilization for vm in self.vms), default=0.0) * 100.0

        attempted = self.placed_tasks + self.failed_tasks
        failure_rate = (self.failed_tasks / attempted * 100.0) if attempted else 0.0

        by_provider: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for vm in self.vms:
            by_provider[vm.provider.value] = by_provider.get(vm.provider.value, 0) + 1
            by_type[vm.type.value] = by_type.get(vm.type.value, 0) + 1

        return {
            "clock": round(self.clock, 1),
            "fleet_size": len(self.vms),
            "cpu_capacity": cap_cpu,
            "ram_capacity": cap_ram,
            "cpu_used": round(use_cpu, 3),
            "ram_used": round(use_ram, 3),
            "cpu_utilization": round(cpu_util, 2),
            "ram_utilization": round(ram_util, 2),
            "mean_utilization": round((cpu_util + ram_util) / 2.0, 2),
            "wastage_percentage": round(wastage, 2),
            "hourly_cost": round(hourly, 4),
            "daily_cost": round(hourly * 24, 2),
            "accrued_cost": round(self.accrued_cost, 4),
            "power_watts": round(watts, 1),
            "co2_kg_per_hour": round(co2_hr, 4),
            "accrued_co2_kg": round(self.accrued_co2_kg, 4),
            "accrued_energy_kwh": round(self.accrued_energy_kwh, 4),
            "sla_compliance": round(sla, 2),
            "sla_violations": violations,
            "provisioning_pressure": round(pressure, 2),
            "tasks_running": sum(len(vm.tasks) for vm in self.vms),
            "tasks_completed": self.completed_tasks,
            "tasks_failed": self.failed_tasks,
            "task_failure_rate": round(failure_rate, 2),
            "by_provider": by_provider,
            "by_instance_type": by_type,
            "mean_latency_ms": round(
                sum(latency_ms(vm.provider, vm.region) for vm in self.vms) / len(self.vms), 1
            ) if self.vms else 0.0,
        }

    def fleet_snapshot(self) -> list[dict]:
        return [vm.as_dict() for vm in self.vms]


# ---------------------------------------------------------------------------
# Autoscalers (baselines for the ablation study)
# ---------------------------------------------------------------------------

class AutoScaler:
    """Fleet sizing without reinforcement learning.

    `reactive`   - scales after measured utilisation crosses a threshold, with a
                   provisioning delay, which is the classic threshold autoscaler.
    `predictive` - scales on the ML forecast for the next interval.
    """

    def __init__(
        self,
        allocator: ResourceAllocator,
        mode: Literal["reactive", "predictive"] = "predictive",
        scale_up_threshold: float = 0.80,
        scale_down_threshold: float = 0.35,
        headroom: float = 1.20,
        provisioning_delay_ticks: int = 2,
        cooldown_ticks: int = 2,
    ):
        self.allocator = allocator
        self.mode = mode
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.headroom = headroom
        self.provisioning_delay_ticks = provisioning_delay_ticks
        self.cooldown_ticks = cooldown_ticks
        self._pending: list[tuple[int, InstanceType]] = []
        self._last_action_tick = -99

    def _commit_pending(self, tick_index: int) -> int:
        """Apply scale-ups whose provisioning delay has elapsed."""
        ready = [(t, k) for (t, k) in self._pending if t <= tick_index]
        self._pending = [(t, k) for (t, k) in self._pending if t > tick_index]
        for _, kind in ready:
            self.allocator.add_vm(kind)
        return len(ready)

    def step(
        self,
        tick_index: int,
        predicted_cpu: float,
        predicted_ram: float,
        observed_cpu: float | None = None,
    ) -> dict:
        """Resize the fleet for the coming interval.

        `observed_cpu` is the CPU actually consumed in the *previous* interval.
        A reactive autoscaler must act on that, because it is the only signal a
        threshold policy has - reading instantaneous usage at the top of a tick
        samples an empty fleet (the previous cohort has already retired) and
        would scale down to the floor every time.
        """
        applied = self._commit_pending(tick_index)
        actions: list[str] = ["provisioned" for _ in range(applied)]

        cap_cpu, cap_ram = self.allocator.capacity()
        live_cpu, _ = self.allocator.usage()
        use_cpu = live_cpu if observed_cpu is None else observed_cpu

        if self.mode == "reactive":
            signal_cpu = use_cpu
            util = (use_cpu / cap_cpu) if cap_cpu else 1.0
            trigger_up = util > self.scale_up_threshold
            trigger_down = util < self.scale_down_threshold
        else:
            signal_cpu = predicted_cpu
            projected = (predicted_cpu / cap_cpu) if cap_cpu else 1.0
            trigger_up = projected > self.scale_up_threshold
            trigger_down = projected < self.scale_down_threshold

        on_cooldown = (tick_index - self._last_action_tick) < self.cooldown_ticks

        if trigger_up and not on_cooldown:
            target_cpu = signal_cpu * self.headroom
            target_ram = (predicted_ram if self.mode == 'predictive' else signal_cpu * 2.4) * self.headroom
            deficit = max(0.0, target_cpu - cap_cpu)
            deficit_ram = max(0.0, target_ram - cap_ram)
            guard = 0
            capacity_left = len(self.allocator.vms) + len(self._pending) < MAX_FLEET
            while (deficit > 0 or deficit_ram > 0) and capacity_left:
                guard += 1
                if guard > MAX_FLEET:
                    break
                kind = pick_instance_type(deficit, deficit_ram)
                if self.mode == "reactive":
                    # Reactive scaling pays a provisioning delay: capacity is
                    # requested now but only arrives some ticks later.
                    self._pending.append((tick_index + self.provisioning_delay_ticks, kind))
                    actions.append(f"requested_{kind.value}")
                else:
                    self.allocator.add_vm(kind)
                    actions.append(f"added_{kind.value}")
                deficit -= INSTANCE_SPECS[kind].cpu
                deficit_ram -= INSTANCE_SPECS[kind].ram
                capacity_left = len(self.allocator.vms) + len(self._pending) < MAX_FLEET
            self._last_action_tick = tick_index

        elif trigger_down and not on_cooldown:
            removed = self.allocator.remove_least_utilised()
            if removed:
                actions.append(f"removed_{removed}")
                self._last_action_tick = tick_index

        return {"actions": actions, "pending": len(self._pending)}


# ---------------------------------------------------------------------------
# RL-driven allocator
# ---------------------------------------------------------------------------

def build_state(allocator: ResourceAllocator, predicted_cpu: float, predicted_ram: float) -> list[float]:
    """Six-dimensional observation handed to the DQN."""
    cap_cpu, cap_ram = allocator.capacity()
    use_cpu, use_ram = allocator.usage()
    eps = 1e-6
    hourly = sum(vm.cost_per_hour for vm in allocator.vms)
    # Perfectly-packed cost per core for a MEDIUM instance, used as the scale.
    reference_cost_per_core = INSTANCE_SPECS[InstanceType.MEDIUM].base_cost_per_hour / \
        INSTANCE_SPECS[InstanceType.MEDIUM].cpu
    cost_per_core = (hourly / max(cap_cpu, eps)) / reference_cost_per_core

    return [
        min(2.0, predicted_cpu / max(cap_cpu, eps)),
        min(2.0, predicted_ram / max(cap_ram, eps)),
        min(1.5, use_cpu / max(cap_cpu, eps)),
        min(1.5, use_ram / max(cap_ram, eps)),
        len(allocator.vms) / MAX_FLEET,
        min(3.0, cost_per_core),
    ]


def compute_reward(
    allocator: ResourceAllocator,
    action: int,
    demand_cpu: float,
    placement_failures: int = 0,
    tasks_submitted: int = 0,
    fleet_changes: int = 0,
) -> tuple[float, dict]:
    """Reward balances utilisation, cost and SLA - the three Sprint II objectives.

    utilisation term  Gaussian centred on TARGET_UTILISATION, so both
                      over-provisioning and under-provisioning lose reward.
    cost term         penalises dollars per unit of served demand above the
                      cost of a perfectly packed fleet.
    SLA term          penalises hot VMs and any task that could not be placed.
    churn term        small penalty on fleet changes to damp oscillation.
    """
    cap_cpu, _ = allocator.capacity()
    use_cpu, _ = allocator.usage()
    util = (use_cpu / cap_cpu) if cap_cpu else 0.0

    # Asymmetric on purpose. A symmetric bell around the target punishes the
    # agent for provisioning the headroom it needs to absorb forecast error and
    # packing fragmentation, which pushes it into dropping work. Idle capacity
    # is penalised here; running hot is left to the hot-node and drop terms,
    # which measure the actual harm rather than a proxy.
    if util >= TARGET_UTILISATION:
        util_term = 1.0
    else:
        util_term = math.exp(-((util - TARGET_UTILISATION) ** 2) / (2 * 0.18**2))

    hourly = sum(vm.cost_per_hour for vm in allocator.vms)
    spec = INSTANCE_SPECS[InstanceType.MEDIUM]
    reference = spec.base_cost_per_hour / spec.cpu
    cost_ratio = (hourly / max(demand_cpu, 1.0)) / reference
    cost_term = max(0.0, cost_ratio - (1.0 / TARGET_UTILISATION))

    violations = allocator._sla_violations()
    hot_term = (violations / len(allocator.vms)) if allocator.vms else 1.0

    # Rejected work is the dominant failure mode and must be scored as a rate,
    # not a count: a fixed per-task penalty saturates after four drops and the
    # agent stops distinguishing "dropped 4" from "dropped 40".
    drop_rate = (placement_failures / tasks_submitted) if tasks_submitted else 0.0

    churn_term = min(1.0, fleet_changes / 6.0)

    # The drop coefficient is deliberately an order of magnitude above the cost
    # coefficient. At 2.5 the measured optimum was a lean fleet that rejected
    # ~13% of tasks, because the cost saved outweighed the penalty - the agent
    # was correctly optimising a badly-specified objective. At 8.0 even a 5%
    # drop rate (-0.40) costs more than any achievable cost saving.
    reward = (
        1.00 * util_term
        - 0.30 * min(3.0, cost_term)
        - 0.40 * hot_term
        - 8.00 * drop_rate
        - 0.05 * churn_term
    )
    breakdown = {
        "utilisation_term": round(util_term, 4),
        "cost_term": round(-0.30 * min(3.0, cost_term), 4),
        "hot_node_term": round(-0.40 * hot_term, 4),
        "drop_term": round(-8.00 * drop_rate, 4),
        "churn_term": round(-0.05 * churn_term, 4),
        "utilisation": round(util * 100, 2),
        "drop_rate": round(drop_rate * 100, 2),
    }
    return float(reward), breakdown


def pick_instance_type(gap_cpu: float, gap_ram: float) -> InstanceType:
    """Choose a node whose shape matches the deficit, not just its size.

    If the memory deficit outweighs the CPU deficit by more than the
    general-purpose RAM:CPU ratio, adding another balanced node strands cores
    behind exhausted memory - a memory-optimised node is the right answer.
    """
    gap_cpu = max(0.0, gap_cpu)
    gap_ram = max(0.0, gap_ram)
    if gap_ram > gap_cpu * 2.6:
        return InstanceType.MEMORY
    # Compare deficits in CPU-equivalent units for the size decision.
    gap = max(gap_cpu, gap_ram / 2.0)
    if gap > 6:
        return InstanceType.LARGE
    if gap > 2.5:
        return InstanceType.MEDIUM
    return InstanceType.SMALL


def apply_action(
    allocator: ResourceAllocator,
    action: int,
    predicted_cpu: float,
    predicted_ram: float = 0.0,
) -> tuple[str, int]:
    """Resize the fleet to the headroom setpoint the agent selected.

    Sizing considers CPU *and* RAM. Sizing on CPU alone is a real trap here:
    the workload's RAM:CPU ratio (~2.4) is higher than a medium node's (2.0),
    so memory saturates first and tasks are rejected while CPU still looks
    comfortable. Capacity must satisfy the binding resource, whichever it is.

    Returns a human-readable label and the number of nodes changed, which the
    reward uses as a churn signal.
    """
    multiplier = HEADROOM_LEVELS[action]
    small = INSTANCE_SPECS[InstanceType.SMALL]
    target_cpu = max(small.cpu, predicted_cpu * multiplier)
    target_ram = max(small.ram, predicted_ram * multiplier)

    cap_cpu, cap_ram = allocator.capacity()
    added = removed = 0

    def satisfied(c: float, r: float) -> bool:
        return c >= target_cpu and r >= target_ram

    guard = 0
    while not satisfied(cap_cpu, cap_ram) and len(allocator.vms) < MAX_FLEET:
        guard += 1
        if guard > MAX_FLEET:
            break
        kind = pick_instance_type(target_cpu - cap_cpu, target_ram - cap_ram)
        vm = allocator.add_vm(kind)
        if vm is None:
            break
        cap_cpu += vm.cpu_capacity
        cap_ram += vm.ram_capacity
        added += 1

    guard = 0
    while len(allocator.vms) > MIN_FLEET:
        guard += 1
        if guard > MAX_FLEET:
            break
        # Retire the least useful node, but only while both resources stay
        # above the setpoint - never shrink below what the forecast calls for.
        candidate = min(allocator.vms, key=lambda v: (v.cpu_utilization, -v.cost_per_hour))
        if not satisfied(cap_cpu - candidate.cpu_capacity, cap_ram - candidate.ram_capacity):
            break
        if not allocator.remove_vm(candidate.id):
            break
        cap_cpu -= candidate.cpu_capacity
        cap_ram -= candidate.ram_capacity
        removed += 1

    label = (f"headroom {multiplier:.2f}x -> {cap_cpu:.0f} cores / {cap_ram:.0f} GB "
             f"(+{added}/-{removed}, {len(allocator.vms)} nodes)")
    return label, added + removed


# ---------------------------------------------------------------------------
# Advisory (US-07, US-13, US-15)
# ---------------------------------------------------------------------------

@dataclass
class Advisory:
    warnings: list[dict]
    recommendations: list[dict]
    potential_hourly_saving: float
    current_utilisation: float
    predicted_utilisation: float

    def as_dict(self) -> dict:
        return {
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "potential_hourly_saving": round(self.potential_hourly_saving, 4),
            "current_utilisation": round(self.current_utilisation, 2),
            "predicted_utilisation": round(self.predicted_utilisation, 2),
        }


class AdvisoryEngine:
    def __init__(self, allocator: ResourceAllocator):
        self.allocator = allocator

    def generate(
        self,
        predicted_cpu: float,
        predicted_ram: float,
        anomaly: dict | None = None,
    ) -> Advisory:
        a = self.allocator
        metrics = a.get_metrics()
        cap_cpu, _ = a.capacity()
        warnings: list[dict] = []
        recommendations: list[dict] = []
        saving = 0.0

        predicted_util = (predicted_cpu / cap_cpu * 100.0) if cap_cpu else 0.0

        # 1. Capacity exhaustion
        if cap_cpu and predicted_cpu > cap_cpu * 0.85:
            critical = predicted_cpu >= cap_cpu
            warnings.append({
                "type": "CAPACITY_CRUNCH",
                "severity": "critical" if critical else "warning",
                "message": (
                    f"Forecast demand of {predicted_cpu:.1f} cores reaches "
                    f"{predicted_util:.0f}% of the {cap_cpu:.0f}-core fleet."
                ),
                "eta": "immediate" if critical else "next 30-60 min",
            })
            deficit = max(0.0, predicted_cpu * 1.2 - cap_cpu)
            nodes = max(1, math.ceil(deficit / INSTANCE_SPECS[InstanceType.MEDIUM].cpu))
            recommendations.append({
                "action": f"Provision {nodes} medium node(s) ahead of the peak",
                "benefit": "Avoids the reactive-scaling latency window and the SLA breach it causes.",
                "urgency": "high",
            })

        # 2. Over-provisioning
        if cap_cpu and predicted_cpu < cap_cpu * 0.35 and len(a.vms) > MIN_FLEET:
            idle_cores = cap_cpu - predicted_cpu * 1.25
            removable = max(0, int(idle_cores // INSTANCE_SPECS[InstanceType.MEDIUM].cpu))
            if removable:
                est = removable * INSTANCE_SPECS[InstanceType.MEDIUM].base_cost_per_hour
                saving += est
                warnings.append({
                    "type": "BUDGET_LEAK",
                    "severity": "optimisation",
                    "message": (
                        f"Forecast demand uses only {predicted_util:.0f}% of provisioned "
                        f"capacity. {idle_cores:.0f} cores are being paid for and not used."
                    ),
                    "eta": "immediate",
                })
                recommendations.append({
                    "action": f"Decommission {removable} idle node(s)",
                    "benefit": f"Saves about ${est:.2f}/hr (${est * 24:.2f}/day).",
                    "urgency": "medium",
                })

        # 3. Multi-cloud arbitrage
        if a.vms:
            board = score_providers(a.clock, InstanceType.MEDIUM, a.region)
            best = board[0]
            current = max(metrics["by_provider"], key=metrics["by_provider"].get)
            current_row = next(r for r in board if r["provider"] == current)
            delta = current_row["hourly_cost"] - best["hourly_cost"]
            if best["provider"] != current and delta > 0:
                pct = delta / current_row["hourly_cost"] * 100.0
                if pct >= 4.0:
                    est = delta * len(a.vms)
                    saving += est
                    warnings.append({
                        "type": "PRICE_ARBITRAGE",
                        "severity": "optimisation",
                        "message": (
                            f"{best['provider']} is {pct:.1f}% cheaper than {current} "
                            f"for medium instances right now."
                        ),
                        "eta": "now",
                    })
                    recommendations.append({
                        "action": f"Route new workloads to {best['provider']}",
                        "benefit": f"About ${est:.2f}/hr at the current fleet size.",
                        "urgency": "medium",
                    })

        # 4. SLA risk
        if metrics["sla_violations"] > 0:
            warnings.append({
                "type": "SLA_RISK",
                "severity": "critical",
                "message": (
                    f"{metrics['sla_violations']} node(s) above "
                    f"{SLA_CRITICAL_UTILISATION * 100:.0f}% CPU. Queueing delay is likely."
                ),
                "eta": "immediate",
            })

        # 5. Anomaly
        if anomaly and anomaly.get("is_anomaly"):
            warnings.append({
                "type": "WORKLOAD_ANOMALY",
                "severity": "warning",
                "message": (
                    f"Demand pattern flagged as anomalous by {anomaly['method']} "
                    f"(severity {anomaly['severity']:.2f})."
                ),
                "eta": "now",
            })
            recommendations.append({
                "action": "Hold current capacity and inspect the workload source",
                "benefit": "Prevents the autoscaler from chasing a spurious spike.",
                "urgency": "medium",
            })

        return Advisory(
            warnings=warnings,
            recommendations=recommendations,
            potential_hourly_saving=saving,
            current_utilisation=metrics["cpu_utilization"],
            predicted_utilisation=predicted_util,
        )


# ---------------------------------------------------------------------------
# SmartAllocator - the integrated brain
# ---------------------------------------------------------------------------

class SmartAllocator(ResourceAllocator):
    """`ResourceAllocator` + trained predictor + anomaly detector + DQN agent.

    Difference from a naive integration: the lag/rolling features handed to the
    predictor at inference time are taken from a live ring buffer of observed
    demand, not hard-coded constants. Feeding the model placeholder lags makes
    every online prediction fall outside the distribution it was trained on.
    """

    HISTORY = 32

    def __init__(
        self,
        predictor_algo: str = "xgboost",
        anomaly_method: str = "isolation_forest",
        artifacts_dir=None,
        region: Region = Region.US_EAST,
        multi_cloud: bool = True,
        seed: int = 42,
    ):
        super().__init__(region=region, multi_cloud=multi_cloud)
        from collections import deque
        from pathlib import Path

        from .ml_models import AnomalyDetector, DQNAgent, WorkloadPredictor

        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else _default_artifacts()
        self.predictor_algo = predictor_algo
        self.anomaly_method = anomaly_method
        self.load_errors: list[str] = []

        try:
            self.predictor = WorkloadPredictor.load(self.artifacts_dir, predictor_algo)
        except Exception as exc:                       # surfaced, never swallowed
            self.predictor = None
            self.load_errors.append(f"predictor({predictor_algo}): {exc}")

        try:
            self.anomaly_detector = AnomalyDetector.load(self.artifacts_dir, anomaly_method)
        except Exception as exc:
            self.anomaly_detector = None
            self.load_errors.append(f"anomaly({anomaly_method}): {exc}")

        self.dqn_agent = DQNAgent(seed=seed)
        self.dqn_loaded = self.dqn_agent.load(self.artifacts_dir / "dqn_agent.json")

        self.advisory = AdvisoryEngine(self)
        self.cpu_history: "deque[float]" = deque(maxlen=self.HISTORY)
        self.ram_history: "deque[float]" = deque(maxlen=self.HISTORY)
        self.last_prediction: dict = {}

    # -- online feature construction -------------------------------------

    def observe(self, cpu_demand: float, ram_demand: float) -> None:
        """Record realised demand so the next prediction has honest lags."""
        self.cpu_history.append(float(cpu_demand))
        self.ram_history.append(float(ram_demand))

    def _lag_features(self) -> dict:
        import numpy as np

        cpu = list(self.cpu_history)
        ram = list(self.ram_history)
        if not cpu:
            # Cold start: fall back to current fleet usage rather than a constant.
            use_cpu, use_ram = self.usage()
            cpu, ram = [use_cpu or 1.0], [use_ram or 1.0]
        return {
            "cpu_lag_1": cpu[-1],
            "cpu_lag_4": cpu[-4] if len(cpu) >= 4 else cpu[0],
            "cpu_rolling_mean_4": float(np.mean(cpu[-4:])),
            "cpu_rolling_std_8": float(np.std(cpu[-8:])) if len(cpu) >= 2 else 0.0,
            "ram_lag_1": ram[-1],
        }

    def build_feature_row(
        self, num_tasks: float, cpu_per_task: float, ram_per_task: float,
        hour: int, day_of_week: int,
    ) -> dict:
        import numpy as np

        return {
            "num_tasks": float(num_tasks),
            "cpu_per_task": float(cpu_per_task),
            "ram_per_task": float(ram_per_task),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
            "day_of_week": float(day_of_week),
            "is_weekend": 1.0 if day_of_week >= 5 else 0.0,
            **self._lag_features(),
        }

    # -- prediction ------------------------------------------------------

    def predict_demand(
        self, num_tasks: float, cpu_per_task: float, ram_per_task: float,
        hour: int, day_of_week: int,
    ) -> tuple[float, float]:
        if self.predictor is None:
            raise RuntimeError(
                "Workload predictor unavailable: " + "; ".join(self.load_errors)
            )
        row = self.build_feature_row(num_tasks, cpu_per_task, ram_per_task, hour, day_of_week)
        cpu, ram = self.predictor.predict(row)
        self.last_prediction = {"features": row, "cpu": cpu, "ram": ram}
        return cpu, ram

    def explain_last(self) -> dict:
        if self.predictor is None or not self.last_prediction:
            return {"method": "unavailable", "contributions": [], "base_value": 0.0}
        return self.predictor.explain(self.last_prediction["features"])

    def check_anomaly(self, cpu_demand: float, ram_demand: float) -> dict | None:
        """Score the current demand against its own recent history.

        The detector needs context, not just the level - a burst is defined by
        being abrupt relative to the trailing window, so the live ring buffer
        supplies the previous value and the rolling means.
        """
        if self.anomaly_detector is None:
            return None
        import numpy as np

        cpu, ram = list(self.cpu_history), list(self.ram_history)
        return self.anomaly_detector.check(
            cpu_demand,
            ram_demand,
            cpu_prev=cpu[-1] if cpu else None,
            cpu_rolling=float(np.mean(cpu[-6:])) if cpu else None,
            ram_rolling=float(np.mean(ram[-6:])) if ram else None,
        )

    # -- RL control loop (state -> action -> reward -> policy update) -----

    # The RL interaction is split in two because the reward for an action is
    # only observable *after* the interval's work has been placed against the
    # resized fleet. Scoring the action immediately would read an empty fleet -
    # zero utilisation, zero failures - and the agent would learn that the
    # cheapest fleet is always best, converging on a one-node cluster that
    # drops most of the workload.

    def rl_begin(
        self, predicted_cpu: float, predicted_ram: float, greedy: bool = False
    ) -> dict:
        """Observe, choose an action, and apply it to the fleet."""
        state = build_state(self, predicted_cpu, predicted_ram)
        action = self.dqn_agent.act(state, greedy=greedy)
        effect, changes = apply_action(self, action, predicted_cpu, predicted_ram)
        return {
            "state": state,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "effect": effect,
            "fleet_changes": changes,
            "q_values": [round(float(q), 4) for q in self.dqn_agent.q_values(state)],
        }

    def rl_complete(
        self,
        pending: dict,
        predicted_cpu: float,
        predicted_ram: float,
        placement_failures: int = 0,
        tasks_submitted: int = 0,
        train: bool = True,
        done: bool = False,
    ) -> dict:
        """Score the action against the realised interval and update the policy."""
        reward, breakdown = compute_reward(
            self, pending["action"], predicted_cpu,
            placement_failures=placement_failures,
            tasks_submitted=tasks_submitted,
            fleet_changes=pending.get("fleet_changes", 0),
        )
        next_state = build_state(self, predicted_cpu, predicted_ram)

        loss = None
        if train:
            self.dqn_agent.remember(
                pending["state"], pending["action"], reward, next_state, done
            )
            loss = self.dqn_agent.learn()

        return {
            "state": [round(v, 4) for v in pending["state"]],
            "action": pending["action"],
            "action_name": pending["action_name"],
            "effect": pending["effect"],
            "reward": round(reward, 4),
            "reward_breakdown": breakdown,
            "epsilon": round(self.dqn_agent.epsilon, 4),
            "loss": None if loss is None else round(loss, 6),
            "q_values": pending["q_values"],
        }

    def status(self) -> dict:
        return {
            "predictor_algo": self.predictor_algo,
            "predictor_ready": self.predictor is not None,
            "anomaly_method": self.anomaly_method,
            "anomaly_ready": self.anomaly_detector is not None,
            "dqn_pretrained": self.dqn_loaded,
            "dqn_epsilon": round(self.dqn_agent.epsilon, 4),
            "dqn_learn_steps": self.dqn_agent.learn_steps,
            "load_errors": self.load_errors,
            "history_depth": len(self.cpu_history),
        }


def _default_artifacts():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "artifacts"


__all__ = [
    "ResourceAllocator", "SmartAllocator", "AutoScaler", "AdvisoryEngine", "Advisory",
    "build_state", "compute_reward", "apply_action",
    "MAX_FLEET", "MIN_FLEET", "TARGET_UTILISATION", "SLA_CRITICAL_UTILISATION",
    "ACTION_NAMES",
]
