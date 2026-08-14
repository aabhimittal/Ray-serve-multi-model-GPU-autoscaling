"""Observability: Prometheus exposition for the serving fleet."""

from ray_serve_autoscale.observability.prometheus import render_prometheus

__all__ = ["render_prometheus"]
