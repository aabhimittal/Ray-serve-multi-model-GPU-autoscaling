"""Runtime configuration.

Settings are resolved with the following precedence (highest first):

1. Environment variables prefixed with ``RSA_`` (e.g. ``RSA_BACKEND=simulated``).
2. Values in the YAML file pointed to by ``RSA_CONFIG_FILE`` (default:
   ``config/models.yaml`` if it exists).
3. The defaults declared on the models below.

Keeping configuration in one typed place means the deployments, the
autoscaler and the CLI all agree on the same knobs.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Backend = Literal["simulated", "transformers"]


class ModelConfig(BaseModel):
    """Per-model deployment configuration."""

    name: str
    task: Literal["sentiment", "summarization", "embedding"]
    hf_model_id: str = ""
    # GPU fraction requested per replica. Fractional GPUs let several small
    # models share one physical card; set to a whole number for large models.
    num_gpus: float = 0.5
    num_cpus: float = 1.0
    # Ray Serve built-in autoscaling bounds.
    min_replicas: int = 1
    max_replicas: int = 6
    # Target concurrent requests per replica. By Little's Law this is the
    # primary lever that trades throughput against tail latency.
    target_ongoing_requests: int = 8
    max_ongoing_requests: int = 16
    # Simulated-backend compute time (seconds) used when torch/transformers
    # are unavailable. Lets CI exercise the latency/autoscaling machinery.
    simulated_latency_s: float = 0.05
    # Per-model latency SLO (milliseconds). The custom autoscaler treats p95
    # above this as pressure to scale out.
    latency_slo_ms: float = 250.0


class AutoscalerConfig(BaseModel):
    """Custom latency-based autoscaling controller configuration."""

    enabled: bool = True
    # How often the controller samples latency and reconsiders replica counts.
    control_interval_s: float = 15.0
    # p95 latency above ``slo * scale_up_ratio`` triggers a scale-out.
    scale_up_ratio: float = 0.9
    # p95 latency below ``slo * scale_down_ratio`` permits a scale-in.
    scale_down_ratio: float = 0.4
    # Replicas added/removed per control step.
    scale_step: int = 1
    # Cooldown after any scaling action before another is allowed.
    cooldown_s: float = 60.0
    # Minimum requests observed in a window before acting (avoids reacting to
    # noise from a near-idle endpoint).
    min_samples: int = 20


class Settings(BaseModel):
    """Top-level application settings."""

    backend: Backend = "simulated"
    ray_address: str | None = None
    http_host: str = "0.0.0.0"
    http_port: int = 8000
    route_prefix: str = "/"
    models: list[ModelConfig] = Field(default_factory=list)
    autoscaler: AutoscalerConfig = Field(default_factory=AutoscalerConfig)

    def model_by_name(self, name: str) -> ModelConfig | None:
        return next((m for m in self.models if m.name == name), None)


def _default_models() -> list[ModelConfig]:
    """Sensible defaults so the app runs out-of-the-box with no config file."""
    return [
        ModelConfig(
            name="sentiment",
            task="sentiment",
            hf_model_id="distilbert-base-uncased-finetuned-sst-2-english",
            num_gpus=0.25,
            min_replicas=1,
            max_replicas=6,
            target_ongoing_requests=8,
            simulated_latency_s=0.03,
            latency_slo_ms=150.0,
        ),
        ModelConfig(
            name="summarization",
            task="summarization",
            hf_model_id="sshleifer/distilbart-cnn-12-6",
            num_gpus=0.5,
            min_replicas=1,
            max_replicas=4,
            target_ongoing_requests=4,
            simulated_latency_s=0.15,
            latency_slo_ms=800.0,
        ),
        ModelConfig(
            name="embedding",
            task="embedding",
            hf_model_id="sentence-transformers/all-MiniLM-L6-v2",
            num_gpus=0.25,
            min_replicas=1,
            max_replicas=6,
            target_ongoing_requests=16,
            simulated_latency_s=0.02,
            latency_slo_ms=120.0,
        ),
    ]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_file: str | None = None) -> Settings:
    """Build a :class:`Settings` object from YAML + environment overrides."""
    path_str = config_file or os.environ.get("RSA_CONFIG_FILE")
    data: dict = {}
    if path_str:
        path = Path(path_str)
        if path.exists():
            data = _load_yaml(path)

    settings = Settings(**data) if data else Settings()
    if not settings.models:
        settings.models = _default_models()

    # Environment overrides for the most commonly tuned knobs.
    if backend := os.environ.get("RSA_BACKEND"):
        settings.backend = backend  # type: ignore[assignment]
    if addr := os.environ.get("RSA_RAY_ADDRESS"):
        settings.ray_address = addr
    if port := os.environ.get("RSA_HTTP_PORT"):
        settings.http_port = int(port)
    if os.environ.get("RSA_AUTOSCALER_ENABLED") is not None:
        settings.autoscaler.enabled = os.environ["RSA_AUTOSCALER_ENABLED"].lower() in (
            "1",
            "true",
            "yes",
        )
    return settings


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor for process-wide settings."""
    return load_settings()
