"""Command-line entrypoint: ``rsa-serve``.

Subcommands
-----------
serve       start Ray, deploy the app, and (optionally) run the latency
            autoscaler control loop in a background thread.
autoscale   run ONLY the latency autoscaler against an already-running gateway
            (useful to run the controller as its own process/pod).
config      print the resolved settings and exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading

from ray_serve_autoscale.settings import Settings, load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ray_serve_autoscale.cli")


def _http_latency_reader(base_url: str):
    """Reader that pulls latency stats from the gateway HTTP endpoint."""
    import requests

    def reader() -> dict[str, dict]:
        resp = requests.get(f"{base_url.rstrip('/')}/metrics/latency", timeout=5)
        resp.raise_for_status()
        return resp.json().get("latency", {})

    return reader


def _start_autoscaler(settings: Settings, base_url: str, dashboard_url: str) -> threading.Thread:
    from ray_serve_autoscale.autoscaling.latency_autoscaler import (
        LatencyAutoscaler,
        LoggingApplier,
        ServeRestApplier,
    )

    applier = ServeRestApplier(dashboard_url) if settings.autoscaler.enabled else LoggingApplier()
    controller = LatencyAutoscaler(
        models=settings.models,
        cfg=settings.autoscaler,
        reader=_http_latency_reader(base_url),
        applier=applier,
    )
    thread = threading.Thread(target=controller.run_forever, name="latency-autoscaler", daemon=True)
    thread.start()
    return thread


def cmd_serve(args: argparse.Namespace) -> int:
    import ray
    from ray import serve

    from ray_serve_autoscale.app import build_app

    settings = load_settings(args.config)
    ray_address = (settings.ray_address or "auto") if args.attach else None
    ray.init(address=ray_address, ignore_reinit_error=True)
    serve.start(http_options={"host": settings.http_host, "port": settings.http_port})

    app = build_app(settings)
    serve.run(app, route_prefix=settings.route_prefix, name="multi_model_gateway")

    base_url = f"http://127.0.0.1:{settings.http_port}"
    logger.info("gateway deployed at %s", base_url)
    logger.info("try:  curl -s -XPOST %s/predict/sentiment -d '{\"text\":\"great\"}'", base_url)

    if settings.autoscaler.enabled and not args.no_autoscaler:
        _start_autoscaler(settings, base_url, args.dashboard_url)
        logger.info("latency autoscaler running (control_interval=%.0fs)",
                    settings.autoscaler.control_interval_s)

    if args.blocking:
        logger.info("Ctrl-C to shut down")
        try:
            import time

            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("shutting down")
            serve.shutdown()
    return 0


def cmd_autoscale(args: argparse.Namespace) -> int:
    from ray_serve_autoscale.autoscaling.latency_autoscaler import (
        LatencyAutoscaler,
        LoggingApplier,
        ServeRestApplier,
    )

    settings = load_settings(args.config)
    applier = LoggingApplier() if args.dry_run else ServeRestApplier(args.dashboard_url)
    controller = LatencyAutoscaler(
        models=settings.models,
        cfg=settings.autoscaler,
        reader=_http_latency_reader(args.base_url),
        applier=applier,
    )
    controller.run_forever()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    print(json.dumps(settings.model_dump(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rsa-serve", description=__doc__)
    p.add_argument("--config", help="path to a models.yaml config file")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="deploy the multi-model app + autoscaler")
    s.add_argument("--attach", action="store_true", help="attach to a running Ray cluster")
    s.add_argument("--blocking", action="store_true", help="block until Ctrl-C")
    s.add_argument("--no-autoscaler", action="store_true", help="skip the latency controller")
    s.add_argument("--dashboard-url", default="http://127.0.0.1:8265")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("autoscale", help="run only the latency autoscaler")
    a.add_argument("--base-url", default="http://127.0.0.1:8000")
    a.add_argument("--dashboard-url", default="http://127.0.0.1:8265")
    a.add_argument("--dry-run", action="store_true", help="log decisions, don't apply")
    a.set_defaults(func=cmd_autoscale)

    c = sub.add_parser("config", help="print resolved settings and exit")
    c.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
