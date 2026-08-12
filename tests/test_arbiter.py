"""Tests for cluster-wide GPU arbitration and the budget governor."""

import pytest

from ray_serve_autoscale.autoscaling.arbiter import (
    arbitrate,
    headroom_replicas,
    required_gpus,
)
from ray_serve_autoscale.autoscaling.planner import ModelPlan
from ray_serve_autoscale.settings import ClusterConfig


def _plan(name, desired, *, priority=100, gpus=0.5, floor=1, urgency=1.0, cost=2.0):
    return ModelPlan(
        name=name,
        current_replicas=floor,
        desired_replicas=desired,
        min_replicas=floor,
        max_replicas=100,
        gpus_per_replica=gpus,
        priority=priority,
        urgency=urgency,
        bound="queue",
        reason="test",
        cost_per_gpu_hour_usd=cost,
    )


# --- uncontended --------------------------------------------------------------


def test_everything_granted_when_capacity_is_ample():
    alloc = arbitrate(
        [_plan("a", 4), _plan("b", 4)], ClusterConfig(total_gpus=8.0)
    )
    assert alloc.grants["a"].granted_replicas == 4
    assert alloc.grants["b"].granted_replicas == 4
    assert not alloc.contended
    assert alloc.allocated_gpus == pytest.approx(4.0)


def test_gpu_arbitration_disabled_when_cluster_has_no_gpus():
    # CPU dev box / CI: refuse to starve the whole fleet just because the
    # operator never declared GPU capacity.
    alloc = arbitrate([_plan("a", 6)], ClusterConfig(total_gpus=0.0))
    assert alloc.grants["a"].granted_replicas == 6
    assert any("arbitration disabled" in n for n in alloc.notes)


def test_cpu_only_deployments_never_contend():
    alloc = arbitrate(
        [_plan("gpu", 4, gpus=0.5), _plan("cpu", 9, gpus=0.0)],
        ClusterConfig(total_gpus=1.0),
    )
    assert alloc.grants["cpu"].granted_replicas == 9
    assert alloc.grants["gpu"].granted_replicas == 2


# --- contention & priority ----------------------------------------------------


def test_scarce_gpus_go_to_the_higher_priority_model():
    alloc = arbitrate(
        [_plan("critical", 4, priority=200), _plan("batch", 4, priority=100)],
        ClusterConfig(total_gpus=2.0),
    )
    assert alloc.grants["critical"].granted_replicas == 3
    assert alloc.grants["batch"].granted_replicas == 1
    assert alloc.contended
    assert alloc.allocated_gpus == pytest.approx(2.0)


def test_priority_inversion_is_structurally_impossible():
    # The batch model is far more urgent (10x over SLO) but lower priority.
    # It must never preempt the critical model's capacity.
    alloc = arbitrate(
        [
            _plan("critical", 4, priority=200, urgency=1.05),
            _plan("batch", 4, priority=100, urgency=10.0),
        ],
        ClusterConfig(total_gpus=2.0),
    )
    assert alloc.grants["critical"].granted_replicas > alloc.grants["batch"].granted_replicas


def test_urgency_orders_models_within_one_priority_class():
    alloc = arbitrate(
        [
            _plan("calm", 4, priority=100, urgency=1.01),
            _plan("burning", 4, priority=100, urgency=5.0),
        ],
        ClusterConfig(total_gpus=2.0),
    )
    assert alloc.grants["burning"].granted_replicas == 3
    assert alloc.grants["calm"].granted_replicas == 1


def test_floors_are_honoured_before_any_surplus():
    # The critical model wants a lot, but batch's floor is an SLO commitment
    # and must survive before critical gets discretionary capacity.
    alloc = arbitrate(
        [_plan("critical", 10, priority=200), _plan("batch", 10, priority=100, floor=2)],
        ClusterConfig(total_gpus=3.0),
    )
    assert alloc.grants["batch"].granted_replicas >= 2
    assert not alloc.grants["batch"].starved


def test_starvation_is_reported_when_floors_do_not_fit():
    alloc = arbitrate(
        [_plan("a", 2, priority=200), _plan("b", 2, priority=100)],
        ClusterConfig(total_gpus=0.5),
    )
    assert alloc.grants["a"].granted_replicas == 1
    assert alloc.grants["b"].granted_replicas == 0
    assert alloc.starved == ["b"]
    assert any("STARVED" in n for n in alloc.notes)


def test_replica_larger_than_the_whole_cluster_is_starved_not_crashed():
    alloc = arbitrate([_plan("huge", 1, gpus=8.0)], ClusterConfig(total_gpus=4.0))
    assert alloc.grants["huge"].granted_replicas == 0
    assert alloc.grants["huge"].starved


def test_headroom_fraction_reserves_capacity():
    alloc = arbitrate([_plan("a", 8)], ClusterConfig(total_gpus=4.0, gpu_headroom_fraction=0.5))
    assert alloc.usable_gpus == pytest.approx(2.0)
    assert alloc.grants["a"].granted_replicas == 4


def test_fractional_gpus_pack_without_float_drift():
    # 0.1 GPU replicas must pack to exactly 10 in 1.0 GPU despite binary
    # floating point (0.1 * 10 != 1.0).
    alloc = arbitrate([_plan("tiny", 10, gpus=0.1)], ClusterConfig(total_gpus=1.0))
    assert alloc.grants["tiny"].granted_replicas == 10


# --- budget governor ----------------------------------------------------------


def test_budget_trims_from_the_least_important_end():
    # 6 replicas at $1/h each = $6/h against a $3/h cap.
    alloc = arbitrate(
        [
            _plan("critical", 4, priority=200, cost=2.0),
            _plan("batch", 2, priority=100, cost=2.0),
        ],
        ClusterConfig(total_gpus=16.0, max_hourly_budget_usd=3.0),
    )
    assert alloc.hourly_cost_usd <= 3.0 + 1e-6
    assert alloc.grants["critical"].granted_replicas > alloc.grants["batch"].granted_replicas
    assert any("budget governor" in n for n in alloc.notes)


def test_budget_escalates_below_floors_only_when_it_must():
    alloc = arbitrate(
        [_plan("a", 2, floor=2, cost=2.0)],
        ClusterConfig(total_gpus=16.0, max_hourly_budget_usd=0.5),
    )
    assert alloc.hourly_cost_usd <= 0.5 + 1e-6
    assert alloc.grants["a"].starved


def test_generous_budget_changes_nothing():
    alloc = arbitrate(
        [_plan("a", 4)], ClusterConfig(total_gpus=16.0, max_hourly_budget_usd=1000.0)
    )
    assert alloc.grants["a"].granted_replicas == 4
    assert not alloc.over_budget


def test_zero_budget_cuts_all_gpu_replicas():
    alloc = arbitrate(
        [_plan("a", 4)], ClusterConfig(total_gpus=16.0, max_hourly_budget_usd=0.0)
    )
    assert alloc.grants["a"].granted_replicas == 0
    assert alloc.hourly_cost_usd == 0.0


def test_free_models_are_untouched_by_the_budget():
    alloc = arbitrate(
        [_plan("free", 5, cost=0.0)],
        ClusterConfig(total_gpus=16.0, max_hourly_budget_usd=0.0),
    )
    assert alloc.grants["free"].granted_replicas == 5


# --- robustness ---------------------------------------------------------------


def test_empty_plan_list_is_safe():
    alloc = arbitrate([], ClusterConfig(total_gpus=8.0))
    assert alloc.grants == {}
    assert alloc.allocated_gpus == 0.0
    assert not alloc.contended


def test_arbitration_is_deterministic():
    plans = [_plan("a", 4, priority=100), _plan("b", 4, priority=100)]
    first = arbitrate(plans, ClusterConfig(total_gpus=2.0))
    second = arbitrate(plans, ClusterConfig(total_gpus=2.0))
    assert {k: v.granted_replicas for k, v in first.grants.items()} == {
        k: v.granted_replicas for k, v in second.grants.items()
    }


def test_negative_cluster_values_are_sanitized():
    alloc = arbitrate([_plan("a", 2)], ClusterConfig(total_gpus=-8.0))
    # Negative capacity sanitizes to 0 -> arbitration disabled, not chaos.
    assert alloc.grants["a"].granted_replicas == 2


def test_utilization_and_helpers():
    alloc = arbitrate([_plan("a", 2, gpus=0.5)], ClusterConfig(total_gpus=4.0))
    assert alloc.gpu_utilization == pytest.approx(0.25)
    assert headroom_replicas(alloc, 0.5) == 6
    assert headroom_replicas(alloc, 0.0) == 0
    assert required_gpus([_plan("a", 2, gpus=0.5)]) == pytest.approx(1.0)


def test_unmet_demand_is_visible_per_model():
    alloc = arbitrate([_plan("a", 10)], ClusterConfig(total_gpus=1.0))
    grant = alloc.grants["a"]
    assert grant.granted_replicas == 2
    assert grant.unmet_replicas == 8
    assert "capped by arbitration" in grant.reason
