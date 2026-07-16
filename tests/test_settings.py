"""Tests for settings loading, defaults and environment overrides."""

import textwrap

from ray_serve_autoscale.settings import load_settings


def test_defaults_populate_three_models(monkeypatch):
    monkeypatch.delenv("RSA_CONFIG_FILE", raising=False)
    s = load_settings()
    names = {m.name for m in s.models}
    assert names == {"sentiment", "summarization", "embedding"}
    assert s.backend == "simulated"


def test_env_overrides(monkeypatch):
    monkeypatch.delenv("RSA_CONFIG_FILE", raising=False)
    monkeypatch.setenv("RSA_BACKEND", "transformers")
    monkeypatch.setenv("RSA_HTTP_PORT", "9001")
    monkeypatch.setenv("RSA_AUTOSCALER_ENABLED", "false")
    s = load_settings()
    assert s.backend == "transformers"
    assert s.http_port == 9001
    assert s.autoscaler.enabled is False


def test_yaml_config_is_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("RSA_BACKEND", raising=False)
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            backend: simulated
            http_port: 8123
            models:
              - name: tiny
                task: sentiment
                num_gpus: 1.0
                max_replicas: 2
                latency_slo_ms: 99.0
            autoscaler:
              control_interval_s: 5.0
              scale_step: 2
            """
        )
    )
    s = load_settings(str(cfg))
    assert s.http_port == 8123
    assert len(s.models) == 1
    assert s.models[0].name == "tiny"
    assert s.models[0].latency_slo_ms == 99.0
    assert s.autoscaler.scale_step == 2


def test_model_by_name():
    s = load_settings()
    assert s.model_by_name("sentiment").task == "sentiment"
    assert s.model_by_name("does-not-exist") is None
