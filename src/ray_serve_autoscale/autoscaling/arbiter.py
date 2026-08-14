"""Cluster-wide GPU arbitration and budget governance.

Per-model planning answers "how much capacity does this model want?". On a real
cluster that question is never independent: GPUs are finite and shared, so the
sum of locally-correct decisions routinely exceeds what exists. Left
unarbitrated, Ray simply leaves the surplus replicas ``PENDING`` forever -- and
*which* model ends up starved is decided by scheduling accident rather than by
what the business considers important. A batch embedding job can quietly
displace the interactive model behind a customer-facing product.

The arbiter makes that allocation explicit and deterministic:

1. **Floors first, by priority.** Every model's ``min_replicas`` is an SLO
   commitment; those are honoured in priority order before anyone gets extra.
2. **Surplus by strict priority class, urgency within class.** Priority
   dominates so a low-priority model can never preempt a higher-priority one
   (priority inversion is structurally impossible), while urgency -- how far
   past its SLO a model already is -- orders models inside the same class.
3. **Budget governor.** A hard $/hour ceiling trims the allocation from the
   least important end, escalating below floors only if the cap cannot
   otherwise be met, and reporting exactly what it cut.

Nothing here mutates a cluster; it is a pure function from plans + limits to an
allocation, which is what makes the awkward cases testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ray_serve_autoscale.autoscaling.planner import ModelPlan
from ray_serve_autoscale.autoscaling.signals import sanitize
from ray_serve_autoscale.settings import ClusterConfig

# Float slack for fractional-GPU accumulation (0.1+0.2 != 0.3 in binary).
_EPS = 1e-9


@dataclass(frozen=True)
class Grant:
    """Final capacity decision for one model after arbitration."""

    name: str
    desired_replicas: int
    granted_replicas: int
    floor_replicas: int
    gpus_per_replica: float
    priority: int
    urgency: float
    reason: str

    @property
    def unmet_replicas(self) -> int:
        return max(self.desired_replicas - self.granted_replicas, 0)

    @property
    def starved(self) -> bool:
        """True when the model was cut below its SLO floor."""
        return self.granted_replicas < self.floor_replicas

    @property
    def gpus(self) -> float:
        return self.granted_replicas * self.gpus_per_replica


@dataclass(frozen=True)
class Allocation:
    """Cluster-wide arbitration outcome plus the diagnostics operators need."""

    grants: dict[str, Grant] = field(default_factory=dict)
    usable_gpus: float = 0.0
    allocated_gpus: float = 0.0
    hourly_cost_usd: float = 0.0
    budget_usd: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def contended(self) -> bool:
        """Whether demand exceeded supply and someone went unserved."""
        return any(g.unmet_replicas > 0 for g in self.grants.values())

    @property
    def starved(self) -> list[str]:
        return sorted(n for n, g in self.grants.items() if g.starved)

    @property
    def gpu_utilization(self) -> float:
        if self.usable_gpus <= 0:
            return 0.0
        return self.allocated_gpus / self.usable_gpus

    @property
    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.hourly_cost_usd > self.budget_usd + 1e-6


def _rank(plan: ModelPlan) -> tuple:
    """Sort key: highest priority class first, then urgency, then name.

    Priority is compared before urgency, which is what makes preemption of a
    high-priority model by a merely-busier low-priority one impossible. The
    trailing name keeps ordering stable and reproducible across ticks.
    """
    return (-plan.priority, -plan.urgency, plan.name)


def _hourly_cost(counts: dict[str, int], plans: dict[str, ModelPlan]) -> float:
    return sum(
        counts[name] * plans[name].gpus_per_replica * plans[name].cost_per_gpu_hour_usd
        for name in counts
    )


def arbitrate(
    plans: list[ModelPlan],
    cluster: ClusterConfig,
) -> Allocation:
    """Resolve competing per-model demands into a feasible allocation."""
    if not plans:
        return Allocation(usable_gpus=0.0, budget_usd=cluster.max_hourly_budget_usd)

    by_name = {p.name: p for p in plans}
    ordered = sorted(plans, key=_rank)
    notes: list[str] = []

    total_gpus = max(sanitize(cluster.total_gpus), 0.0)
    headroom = min(max(sanitize(cluster.gpu_headroom_fraction), 0.0), 0.95)
    usable = total_gpus * (1.0 - headroom)

    # GPU-free deployments (CPU models, the ingress) never contend for
    # accelerators, so they are granted outright and excluded from the ledger.
    granted: dict[str, int] = {}
    for plan in ordered:
        if plan.gpus_per_replica <= _EPS:
            granted[plan.name] = plan.desired_replicas

    gpu_plans = [p for p in ordered if p.gpus_per_replica > _EPS]

    # total_gpus == 0 means "GPU arbitration disabled" (CPU dev box / CI):
    # honour every plan as-is rather than starving the whole fleet.
    if total_gpus <= _EPS:
        for plan in gpu_plans:
            granted[plan.name] = plan.desired_replicas
        if gpu_plans:
            notes.append("gpu arbitration disabled (cluster.total_gpus=0); granting all plans")
    else:
        remaining = usable

        # -- Phase 1: honour SLO floors, highest priority first -------------
        for plan in gpu_plans:
            floor = min(plan.min_replicas, plan.desired_replicas)
            affordable = int((remaining + _EPS) // plan.gpus_per_replica)
            take = max(min(floor, affordable), 0)
            granted[plan.name] = take
            remaining -= take * plan.gpus_per_replica
            if take < floor:
                notes.append(
                    f"{plan.name}: STARVED below floor "
                    f"({take}/{plan.min_replicas}) -- cluster GPU exhausted"
                )

        # -- Phase 2: distribute the surplus, strict priority classes -------
        # One replica at a time so a single greedy model cannot claim a block
        # of capacity that a higher-priority model needed a slice of.
        progress = True
        while remaining > _EPS and progress:
            progress = False
            for plan in gpu_plans:
                if granted[plan.name] >= plan.desired_replicas:
                    continue
                if plan.gpus_per_replica <= remaining + _EPS:
                    granted[plan.name] += 1
                    remaining -= plan.gpus_per_replica
                    progress = True
                    break  # re-rank from the top after every grant

        if any(granted[p.name] < p.desired_replicas for p in gpu_plans):
            short = ", ".join(
                f"{p.name} {granted[p.name]}/{p.desired_replicas}"
                for p in gpu_plans
                if granted[p.name] < p.desired_replicas
            )
            notes.append(f"GPU contention: unmet demand ({short})")

    # -- Phase 3: budget governor ------------------------------------------
    budget = cluster.max_hourly_budget_usd
    if budget is not None:
        budget = max(sanitize(budget), 0.0)
        cost = _hourly_cost(granted, by_name)
        if cost > budget + 1e-6:
            # Trim from the least important end. Cut discretionary capacity
            # (above floor) everywhere first; only then dip below floors,
            # which is an explicit SLO sacrifice and is reported as such.
            for allow_below_floor in (False, True):
                for plan in reversed(gpu_plans):
                    floor = min(plan.min_replicas, plan.desired_replicas)
                    limit = 0 if allow_below_floor else floor
                    while (
                        granted[plan.name] > limit
                        and _hourly_cost(granted, by_name) > budget + 1e-6
                    ):
                        granted[plan.name] -= 1
                if _hourly_cost(granted, by_name) <= budget + 1e-6:
                    break
            final = _hourly_cost(granted, by_name)
            notes.append(
                f"budget governor: trimmed to ${final:.2f}/h against ${budget:.2f}/h cap"
            )
            if final > budget + 1e-6:
                notes.append("budget cap unreachable even at zero GPU replicas")

    grants: dict[str, Grant] = {}
    for plan in ordered:
        count = granted.get(plan.name, 0)
        if count >= plan.desired_replicas:
            reason = plan.reason
        elif count < min(plan.min_replicas, plan.desired_replicas):
            reason = f"starved below floor by arbitration ({count}/{plan.min_replicas})"
        else:
            reason = f"capped by arbitration at {count}/{plan.desired_replicas}"
        grants[plan.name] = Grant(
            name=plan.name,
            desired_replicas=plan.desired_replicas,
            granted_replicas=count,
            floor_replicas=min(plan.min_replicas, plan.desired_replicas),
            gpus_per_replica=plan.gpus_per_replica,
            priority=plan.priority,
            urgency=plan.urgency,
            reason=reason,
        )

    allocated = sum(g.gpus for g in grants.values())
    return Allocation(
        grants=grants,
        usable_gpus=usable,
        allocated_gpus=allocated,
        hourly_cost_usd=_hourly_cost(granted, by_name),
        budget_usd=cluster.max_hourly_budget_usd,
        notes=notes,
    )


def required_gpus(plans: list[ModelPlan]) -> float:
    """Total GPU the un-arbitrated plans would consume (for reporting)."""
    return sum(p.gpu_demand for p in plans)


def headroom_replicas(allocation: Allocation, gpus_per_replica: float) -> int:
    """How many more replicas of a given size would still fit."""
    if gpus_per_replica <= _EPS:
        return 0
    free = allocation.usable_gpus - allocation.allocated_gpus
    return max(int(math.floor((free + _EPS) / gpus_per_replica)), 0)
