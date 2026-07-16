"""Latency-driven autoscaling for GPU model deployments."""

from ray_serve_autoscale.autoscaling.latency_autoscaler import LatencyAutoscaler
from ray_serve_autoscale.autoscaling.metrics import LatencyWindow

__all__ = ["LatencyAutoscaler", "LatencyWindow"]
