"""Latency-driven autoscaling for GPU model deployments.

Two controllers ship here, layered rather than competing:

* :class:`LatencyAutoscaler` -- per-model, adjusts one deployment's replica
  floor from its own p95. Simple and sufficient on an uncontended cluster.
* :class:`FleetAutoscaler` -- whole-fleet, adds predictive cold-start-aware
  sizing, queue-vs-compute decomposition, priority-class GPU arbitration and a
  budget ceiling. Use it when GPUs are finite and shared.
"""

from ray_serve_autoscale.autoscaling.arbiter import Allocation, Grant, arbitrate
from ray_serve_autoscale.autoscaling.controller import FleetAutoscaler, TickResult
from ray_serve_autoscale.autoscaling.latency_autoscaler import LatencyAutoscaler
from ray_serve_autoscale.autoscaling.metrics import LatencyWindow
from ray_serve_autoscale.autoscaling.planner import (
    CapacityState,
    LatencyBreakdown,
    ModelPlan,
    plan_model,
)
from ray_serve_autoscale.autoscaling.signals import ConditionedSignal, SignalTracker

__all__ = [
    "Allocation",
    "CapacityState",
    "ConditionedSignal",
    "FleetAutoscaler",
    "Grant",
    "LatencyAutoscaler",
    "LatencyBreakdown",
    "LatencyWindow",
    "ModelPlan",
    "SignalTracker",
    "TickResult",
    "arbitrate",
    "plan_model",
]
