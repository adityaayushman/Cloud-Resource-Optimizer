"""Allocator, autoscaler, catalogue and multi-cloud behaviour."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import INSTANCE_SPECS, price_index, score_providers, select_provider
from app.engine import (
    MAX_FLEET,
    AdvisoryEngine,
    AutoScaler,
    ResourceAllocator,
    apply_action,
    build_state,
    compute_reward,
    pick_instance_type,
)
from app.models import CloudProvider, InstanceType, Region, Task, TaskStatus


def make_allocator(n=3, multi_cloud=False):
    alloc = ResourceAllocator(multi_cloud=multi_cloud)
    for _ in range(n):
        alloc.add_vm(InstanceType.MEDIUM, provider=CloudProvider.AWS)
    return alloc


# --------------------------------------------------------------- allocation

def test_task_is_placed_and_consumes_capacity():
    alloc = make_allocator(1)
    assert alloc.allocate_task(Task(1.0, 2.0)) is True
    vm = alloc.vms[0]
    assert vm.cpu_usage == pytest.approx(1.0)
    assert vm.ram_usage == pytest.approx(2.0)
    assert vm.cpu_available == pytest.approx(3.0)


def test_task_that_cannot_fit_is_marked_failed():
    alloc = make_allocator(1)
    task = Task(99.0, 99.0)
    assert alloc.allocate_task(task) is False
    assert task.status is TaskStatus.FAILED
    assert alloc.failed_tasks == 1


def test_best_fit_leaves_least_slack():
    alloc = ResourceAllocator(multi_cloud=False)
    alloc.add_vm(InstanceType.LARGE, provider=CloudProvider.AWS)
    small = alloc.add_vm(InstanceType.SMALL, provider=CloudProvider.AWS)
    alloc.allocate_task(Task(1.5, 3.0), strategy="best_fit")
    # The small node leaves 0.5 CPU spare; the large one would leave 6.5.
    assert small.cpu_usage == pytest.approx(1.5)


def test_utilisation_never_exceeds_capacity():
    alloc = make_allocator(2)
    for _ in range(50):
        alloc.allocate_task(Task(0.4, 1.0))
    for vm in alloc.vms:
        assert vm.cpu_usage <= vm.cpu_capacity + 1e-9
        assert vm.ram_usage <= vm.ram_capacity + 1e-9


# ------------------------------------------------------------ fault recovery

def test_simulate_fault_reallocates_tasks_when_capacity_exists():
    """US-15: killing a node must not lose work if the fleet can absorb it."""
    alloc = make_allocator(3)
    for _ in range(4):
        alloc.allocate_task(Task(0.4, 0.8))
    victim = next(vm for vm in alloc.vms if vm.tasks)
    result = alloc.simulate_fault(victim.id)

    assert result["ok"] is True
    assert result["tasks_lost"] == 0
    assert result["tasks_recovered"] == result["tasks_displaced"]
    assert len(alloc.vms) == 2


def test_remove_vm_refuses_to_drop_below_floor():
    alloc = make_allocator(1)
    assert alloc.remove_vm(alloc.vms[0].id) is False
    assert len(alloc.vms) == 1


# ------------------------------------------------------------------- limits

def test_fleet_is_capped():
    """An unbounded scale-up loop is a billing incident, not a feature."""
    alloc = ResourceAllocator(multi_cloud=False)
    for _ in range(MAX_FLEET + 15):
        alloc.add_vm(InstanceType.SMALL, provider=CloudProvider.AWS)
    assert len(alloc.vms) == MAX_FLEET
    assert alloc.add_vm(InstanceType.SMALL) is None


# ------------------------------------------------------------------ metrics

def test_metrics_expose_every_documented_key():
    alloc = make_allocator(2)
    alloc.allocate_task(Task(1.0, 2.0))
    m = alloc.get_metrics()
    for key in (
        "cpu_utilization", "ram_utilization", "wastage_percentage",
        "hourly_cost", "power_watts", "co2_kg_per_hour", "sla_compliance",
        "provisioning_pressure", "task_failure_rate", "by_provider",
        "by_instance_type", "fleet_size",
    ):
        assert key in m, f"missing metric: {key}"
    assert 0 <= m["cpu_utilization"] <= 100
    assert 0 <= m["sla_compliance"] <= 100
    assert m["power_watts"] > 0


def test_idle_fleet_still_draws_power():
    """A 40% idle floor is the point of the power model - it must not be zero."""
    alloc = make_allocator(2)
    assert alloc.get_metrics()["power_watts"] > 0


def test_tick_accrues_cost_and_retires_finished_tasks():
    alloc = make_allocator(1)
    alloc.allocate_task(Task(1.0, 2.0, duration=300))
    assert alloc.vms[0].cpu_usage > 0
    alloc.tick(300)
    assert alloc.accrued_cost > 0
    assert alloc.accrued_energy_kwh > 0
    assert alloc.vms[0].cpu_usage == pytest.approx(0.0)
    assert alloc.completed_tasks == 1


# -------------------------------------------------------------- multi-cloud

def test_provider_pricing_is_reproducible():
    """Same timestamp, same price - the study must be reproducible."""
    for provider in CloudProvider:
        assert price_index(provider, 1000.0) == price_index(provider, 1000.0)
        assert price_index(provider, 987654.0) == price_index(provider, 987654.0)


def test_provider_pricing_drifts_smoothly_rather_than_jumping():
    """The recommendation must not reshuffle between two page loads.

    Prices are not bitwise constant inside a bucket - a continuous diurnal term
    keeps the series smooth - but they must move by a negligible amount over the
    span of a refresh.
    """
    a = price_index(CloudProvider.AWS, 1000.0)
    b = price_index(CloudProvider.AWS, 1005.0)
    assert a == pytest.approx(b, rel=1e-3)


def test_price_bands_overlap_so_the_cheapest_provider_changes():
    """If one provider were always cheapest, arbitrage would be a no-op."""
    winners = set()
    for i in range(2000):
        rows = score_providers(i * 300.0)
        winners.add(rows[0]["provider"])
    assert len(winners) >= 2


def test_provider_scoreboard_is_ranked_and_complete():
    rows = score_providers(0.0)
    assert len(rows) == 3
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["score"] <= rows[-1]["score"]


def test_multi_cloud_allocator_uses_more_than_one_provider():
    """US-18: load must actually be distributable across providers.

    Sampled across a week of 5-minute price buckets. A short hourly sample can
    legitimately land on one provider for every draw - concentrating on the best
    provider is correct behaviour, not a bug - so the property under test is
    that the selection *responds* to price movement over time.
    """
    alloc = ResourceAllocator(multi_cloud=True)
    for i in range(0, 2016, 6):          # a week, every 30 minutes
        alloc.clock = i * 300.0
        if len(alloc.vms) >= 30:
            alloc.vms.clear()            # stay under the fleet cap
        alloc.add_vm(InstanceType.MEDIUM)
        if len(alloc.get_metrics()["by_provider"]) >= 2:
            return
    pytest.fail("multi-cloud selection never chose a second provider")


def test_selection_weights_change_the_winner():
    """Each criterion must be able to decide the outcome on its own."""
    from app.catalog import SelectionWeights

    latency_pick = select_provider(0.0, weights=SelectionWeights(0, 1, 0))[0]
    carbon_pick = select_provider(0.0, weights=SelectionWeights(0, 0, 1))[0]
    # AWS has the lowest latency; GCP has the lowest PUE. If these agreed, one
    # of the two criteria would be doing no work.
    assert latency_pick is CloudProvider.AWS
    assert carbon_pick is CloudProvider.GCP
    assert latency_pick is not carbon_pick


def test_single_cloud_allocator_pins_to_aws():
    alloc = ResourceAllocator(multi_cloud=False)
    for _ in range(5):
        alloc.add_vm(InstanceType.MEDIUM)
    assert set(alloc.get_metrics()["by_provider"]) == {"AWS"}


# --------------------------------------------------------- instance shaping

def test_memory_deficit_selects_a_memory_optimised_node():
    assert pick_instance_type(1.0, 12.0) is InstanceType.MEMORY
    assert INSTANCE_SPECS[InstanceType.MEMORY].ram / INSTANCE_SPECS[InstanceType.MEMORY].cpu == 4.0


def test_balanced_deficit_selects_general_purpose():
    assert pick_instance_type(8.0, 16.0) is InstanceType.LARGE
    assert pick_instance_type(3.0, 6.0) is InstanceType.MEDIUM
    assert pick_instance_type(1.0, 2.0) is InstanceType.SMALL


# ------------------------------------------------------------------ RL glue

def test_state_vector_shape_and_bounds():
    alloc = make_allocator(3)
    state = build_state(alloc, 10.0, 20.0)
    assert len(state) == 6
    assert all(isinstance(v, float) and v >= 0 for v in state)


def test_every_action_changes_the_fleet_toward_its_setpoint():
    """The action must have a real effect - a cosmetic action cannot be learned."""
    sizes = []
    for action in range(5):
        alloc = make_allocator(2)
        apply_action(alloc, action, predicted_cpu=20.0, predicted_ram=40.0)
        sizes.append(sum(vm.cpu_capacity for vm in alloc.vms))
    # Higher headroom setpoints must provision at least as much capacity.
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_dropping_work_dominates_the_reward():
    """Rejecting tasks must never be the cheaper option."""
    alloc = make_allocator(3)
    for _ in range(10):
        alloc.allocate_task(Task(0.4, 0.8))
    clean, _ = compute_reward(alloc, 1, 10.0, placement_failures=0, tasks_submitted=40)
    dropped, _ = compute_reward(alloc, 1, 10.0, placement_failures=8, tasks_submitted=40)
    assert dropped < clean - 1.0


def test_reward_penalises_idle_over_provisioning():
    lean = make_allocator(3)
    for _ in range(20):
        lean.allocate_task(Task(0.4, 0.8))
    idle = make_allocator(12)
    for _ in range(20):
        idle.allocate_task(Task(0.4, 0.8))

    r_lean, _ = compute_reward(lean, 0, 8.0, tasks_submitted=20)
    r_idle, _ = compute_reward(idle, 0, 8.0, tasks_submitted=20)
    assert r_lean > r_idle


# --------------------------------------------------------------- autoscaler

def test_predictive_autoscaler_provisions_ahead_of_demand():
    alloc = make_allocator(1)
    scaler = AutoScaler(alloc, mode="predictive")
    before = sum(vm.cpu_capacity for vm in alloc.vms)
    scaler.step(0, predicted_cpu=24.0, predicted_ram=48.0)
    assert sum(vm.cpu_capacity for vm in alloc.vms) > before


def test_reactive_autoscaler_defers_capacity_by_its_provisioning_delay():
    """Reactive scaling requests capacity now and receives it later."""
    alloc = make_allocator(1)
    scaler = AutoScaler(alloc, mode="reactive", provisioning_delay_ticks=2)
    before = len(alloc.vms)
    scaler.step(0, predicted_cpu=0.0, predicted_ram=0.0, observed_cpu=3.9)
    assert len(alloc.vms) == before          # nothing yet
    scaler.step(2, predicted_cpu=0.0, predicted_ram=0.0, observed_cpu=3.9)
    assert len(alloc.vms) > before           # arrived


# ----------------------------------------------------------------- advisory

def test_capacity_crunch_warning_fires_when_forecast_exceeds_fleet():
    alloc = make_allocator(1)
    advisory = AdvisoryEngine(alloc).generate(predicted_cpu=40.0, predicted_ram=80.0)
    assert any(w["type"] == "CAPACITY_CRUNCH" for w in advisory.warnings)
    assert advisory.recommendations


def test_budget_leak_warning_fires_when_fleet_is_idle():
    alloc = make_allocator(10)
    advisory = AdvisoryEngine(alloc).generate(predicted_cpu=1.0, predicted_ram=2.0)
    assert any(w["type"] == "BUDGET_LEAK" for w in advisory.warnings)
    assert advisory.potential_hourly_saving > 0


def test_no_advisory_noise_on_a_well_sized_fleet():
    alloc = make_allocator(3)
    advisory = AdvisoryEngine(alloc).generate(predicted_cpu=7.0, predicted_ram=14.0)
    assert not any(w["type"] in ("CAPACITY_CRUNCH", "BUDGET_LEAK") for w in advisory.warnings)


# ------------------------------------------------- real-trace replay support

def test_fleet_cap_is_per_allocator_not_global():
    """A production trace needs a larger ceiling than a small synthetic fleet."""
    small = ResourceAllocator(multi_cloud=False, max_fleet=3)
    for _ in range(10):
        small.add_vm(InstanceType.SMALL, provider=CloudProvider.AWS)
    assert len(small.vms) == 3

    big = ResourceAllocator(multi_cloud=False, max_fleet=25)
    for _ in range(40):
        big.add_vm(InstanceType.SMALL, provider=CloudProvider.AWS)
    assert len(big.vms) == 25


def test_xlarge_is_selected_for_a_large_compute_deficit():
    assert pick_instance_type(40.0, 80.0) is InstanceType.XLARGE
    assert INSTANCE_SPECS[InstanceType.XLARGE].cpu == 16


def test_action_respects_a_raised_cap():
    alloc = ResourceAllocator(multi_cloud=False, max_fleet=60)
    alloc.add_vm(InstanceType.MEDIUM, provider=CloudProvider.AWS)
    apply_action(alloc, 4, predicted_cpu=250.0, predicted_ram=320.0)
    cap = sum(v.cpu_capacity for v in alloc.vms)
    assert cap >= 250.0, f"only provisioned {cap} cores for a 250-core forecast"
    assert len(alloc.vms) <= 60
