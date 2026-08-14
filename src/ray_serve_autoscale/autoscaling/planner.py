"""Per-model capacity planning: predictive, cold-start aware, latency-decomposed.

Two ideas separate this from a step-wise reactive autoscaler.

**1. Predict across the cold-start horizon.**
A GPU replica is not capacity the moment you ask for it -- weights load, a CUDA
context initialises, and 30-60s later it starts serving. A reactive controller
that orders replicas when p95 crosses the SLO is therefore *guaranteed* to
breach the SLO for the whole cold-start window. The planner instead projects
latency forward by ``cold_start_s`` along the measured trend and sizes capacity
for the latency it will *face*, not the latency it already missed.

**2. Decompose latency into queueing and compute.**
Total service time splits into time waiting for a replica and time computing on
one. Only the queueing part responds to replica count:

* **queue-bound** (waiting dominates) -- adding replicas directly cuts p95;
  scale out, aggressively if needed.
* **compute-bound** (the model itself got slower: longer prompts, bigger
  batches, a colder cache) -- more replicas add throughput but barely move
  per-request tail latency. Scaling out here burns GPUs on a problem replicas
  cannot fix, so the planner caps the step and reports the real cause.

Sizing uses a ratio controller (``desired = current x observed/target``) rather
than +/-1 steps, so a fleet that is 4x over its SLO gets 4x the capacity in one
decision instead of crawling there while burning error budget.

Everything here is a pure function of observed state -- no cluster required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ray_serve_autoscale.autoscaling.signals import ConditionedSignal, sanitize
from ray_serve_autoscale.settings import AutoscalerConfig, ModelConfig

# What is limiting latency right now.
BOUND_QUEUE = "queue"
BOUND_COMPUTE = "compute"
BOUND_IDLE = "idle"
BOUND_UNKNOWN = "unknown"


@dataclass(frozen=True)
class LatencyBreakdown:
    """Split of end-to-end latency into waiting vs. computing."""

    total_ms: float
    queue_ms: float
    compute_ms: float

    @classmethod
    def build(cls, total_ms: object, queue_ms: object, compute_ms: object) -> LatencyBreakdown:
        total = sanitize(total_ms)
        queue = sanitize(queue_ms)
        compute = sanitize(compute_ms)
        # Exporters disagree; reconcile rather than trust. If only the parts are
        # known, total is their sum; if only the total is known, we cannot
        # attribute it and leave the parts at zero (-> BOUND_UNKNOWN).
        if total <= 0 and (queue > 0 or compute > 0):
            total = queue + compute
        return cls(total_ms=total, queue_ms=queue, compute_ms=compute)

    @property
    def known(self) -> bool:
        return self.total_ms > 0 and (self.queue_ms > 0 or self.compute_ms > 0)

    @property
    def queue_share(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return min(max(self.queue_ms / self.total_ms, 0.0), 1.0)

    def classify(self, compute_bound_queue_ratio: float) -> str:
        if not self.known:
            return BOUND_UNKNOWN
        return BOUND_COMPUTE if self.queue_share < compute_bound_queue_ratio else BOUND_QUEUE


@dataclass(frozen=True)
class CapacityState:
    """What the fleet actually looks like right now for one model.

    ``pending`` replicas are already ordered but still cold-starting. Counting
    them as capacity-on-the-way is what stops the controller from re-ordering
    the same replicas every tick and overshooting the whole cluster -- the
    classic autoscaler thundering herd.
    """

    ready_replicas: int = 1
    pending_replicas: int = 0
    unhealthy_replicas: int = 0
    inflight_requests: int = 0
    # Replica floor the controller currently has in force. During a total
    # outage there is nothing to measure, so this is the only record of how
    # much capacity the fleet had decided this model needs.
    configured_replicas: int = 0

    @property
    def serving_replicas(self) -> int:
        """Replicas actually able to serve (unhealthy ones are not capacity)."""
        return max(self.ready_replicas - self.unhealthy_replicas, 0)

    @property
    def effective_replicas(self) -> int:
        """Serving replicas plus those already on the way."""
        return self.serving_replicas + max(self.pending_replicas, 0)


@dataclass(frozen=True)
class ModelPlan:
    """Desired capacity for one model, before cluster-wide arbitration."""

    name: str
    current_replicas: int
    desired_replicas: int
    min_replicas: int
    max_replicas: int
    gpus_per_replica: float
    priority: int
    urgency: float
    bound: str
    reason: str
    cost_per_gpu_hour_usd: float = 0.0

    @property
    def direction(self) -> int:
        if self.desired_replicas > self.current_replicas:
            return 1
        if self.desired_replicas < self.current_replicas:
            return -1
        return 0

    @property
    def gpu_demand(self) -> float:
        return self.desired_replicas * self.gpus_per_replica


def _clamp_step(current: int, desired: int, cfg: AutoscalerConfig) -> int:
    """Limit how far one control step may move, in either direction."""
    if desired > current:
        return min(desired, current + max(cfg.max_scale_out_step, 1))
    if desired < current:
        return max(desired, current - max(cfg.max_scale_in_step, 1))
    return desired


def plan_model(
    model: ModelConfig,
    signal: ConditionedSignal,
    capacity: CapacityState,
    cfg: AutoscalerConfig,
    breakdown: Optional[LatencyBreakdown] = None,
    *,
    deadband_multiplier: float = 1.0,
) -> ModelPlan:
    """Compute the replica count this model *wants*, ignoring cluster limits.

    Cluster capacity and budget are deliberately not considered here -- the
    arbiter resolves competing demands globally. Mixing the two would let a
    low-priority model's local decision silently starve a critical one.
    """
    current = capacity.effective_replicas
    floor = max(model.min_replicas, 1)
    ceiling = max(model.max_replicas, floor)
    slo = sanitize(model.latency_slo_ms)

    def _plan(desired: int, urgency: float, bound: str, reason: str) -> ModelPlan:
        bounded = min(max(desired, floor), ceiling)
        # Never scale below what in-flight work needs, or we strand requests
        # mid-flight during a drain.
        needed_for_inflight = math.ceil(
            capacity.inflight_requests / max(model.max_ongoing_requests, 1)
        )
        bounded = max(bounded, min(needed_for_inflight, ceiling))
        return ModelPlan(
            name=model.name,
            current_replicas=current,
            desired_replicas=bounded,
            min_replicas=floor,
            max_replicas=ceiling,
            gpus_per_replica=max(sanitize(model.num_gpus), 0.0),
            priority=model.priority,
            urgency=urgency,
            bound=bound,
            reason=reason,
            cost_per_gpu_hour_usd=max(sanitize(model.cost_per_gpu_hour_usd), 0.0),
        )

    # --- Guards ----------------------------------------------------------
    # A model with no usable SLO cannot be latency-scaled; hold at its floor.
    if slo <= 0:
        return _plan(max(current, floor), 0.0, BOUND_UNKNOWN, "no latency SLO configured")

    # Every replica is gone or unhealthy: restore the floor immediately. This
    # is the one case that must bypass every dead band and sample threshold --
    # there is no traffic *because* there is nothing to serve it.
    if capacity.serving_replicas <= 0 and capacity.pending_replicas <= 0:
        # Restore at least the capacity the fleet had already decided this
        # model needs -- dropping to the bare floor after a crash would make
        # a recovering model immediately breach its SLO again.
        restore = max(floor, capacity.configured_replicas)
        return _plan(restore, 1.0, BOUND_UNKNOWN, "no healthy replicas; restoring floor")

    if not signal.trustworthy:
        why = "stale metrics feed" if signal.is_stale else "no samples in window"
        return _plan(max(current, floor), 0.0, BOUND_IDLE, f"hold: {why}")

    if signal.sample_count < cfg.min_samples:
        return _plan(
            max(current, floor),
            0.0,
            BOUND_IDLE,
            f"hold: {signal.sample_count} samples < {cfg.min_samples}",
        )

    # --- Signal selection -------------------------------------------------
    # Predictive mode looks a cold start ahead; reactive mode uses the
    # smoothed present. Either way we never act on a single raw sample.
    observed = signal.projected_p95_ms if cfg.predictive else signal.smoothed_p95_ms
    urgency = observed / slo if slo > 0 else 0.0

    band = max(deadband_multiplier, 1.0)
    # Widening the band pushes the scale-out trigger up and the scale-in
    # trigger down, so a flapping model needs a decisive move to act on.
    upper = slo * min(cfg.scale_up_ratio * band, 1.0)
    lower = slo * (cfg.scale_down_ratio / band)

    bd = breakdown or LatencyBreakdown.build(0, 0, 0)
    bound = bd.classify(cfg.compute_bound_queue_ratio)

    # The ratio must be taken against the capacity that actually *produced*
    # this measurement -- the serving replicas. Replicas still cold-starting
    # have not influenced latency yet, so folding them into the ratio would
    # re-order the same capacity every tick and overshoot the whole cluster
    # (the classic autoscaler thundering herd).
    measured_on = capacity.serving_replicas or current

    # --- Scale out --------------------------------------------------------
    if observed >= upper:
        target = upper if upper > 0 else slo
        ratio = observed / target if target > 0 else 1.0
        desired = math.ceil(measured_on * ratio) if measured_on > 0 else floor
        # Never order below what is already on the way.
        desired = max(desired, current)

        if bound == BOUND_COMPUTE:
            # Replicas cannot shrink compute time; add at most one for
            # throughput headroom and surface the real cause.
            desired = min(desired, current + 1)
            reason = (
                f"compute-bound (queue {bd.queue_share:.0%} of {bd.total_ms:.0f}ms); "
                f"p95 {observed:.0f}ms >= {upper:.0f}ms -> capped scale-out to {desired}"
            )
        else:
            reason = (
                f"p95 {observed:.0f}ms >= {upper:.0f}ms (x{ratio:.2f}) -> scale out to {desired}"
            )
            if capacity.pending_replicas > 0:
                reason += f"; {capacity.pending_replicas} already warming"
            if cfg.predictive and signal.trend_ms_per_s > 0:
                reason += f"; trend +{signal.trend_ms_per_s:.1f}ms/s"

        return _plan(_clamp_step(current, desired, cfg), urgency, bound, reason)

    # --- Scale in ---------------------------------------------------------
    if observed <= lower:
        # Releasing capacity while replicas are still warming would cancel
        # work already paid for, and the low reading may simply reflect the
        # traffic those replicas have not yet been given.
        if capacity.pending_replicas > 0:
            return _plan(
                current,
                urgency,
                bound,
                f"hold: {capacity.pending_replicas} replicas still warming",
            )
        # Size down proportionally, but only toward the latency we target --
        # never toward the scale-in trigger itself, which would immediately
        # re-breach and start a flap.
        target = slo * cfg.scale_up_ratio
        ratio = observed / target if target > 0 else 1.0
        desired = max(math.ceil(measured_on * ratio), floor) if measured_on > 0 else floor
        desired = min(desired, current)
        reason = f"p95 {observed:.0f}ms <= {lower:.0f}ms -> scale in to {desired}"
        return _plan(_clamp_step(current, desired, cfg), urgency, bound, reason)

    # --- Hold -------------------------------------------------------------
    return _plan(
        max(current, floor),
        urgency,
        bound,
        f"hold: p95 {observed:.0f}ms within [{lower:.0f}, {upper:.0f}]ms",
    )
