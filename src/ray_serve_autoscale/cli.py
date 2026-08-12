"""Command-line entrypoint: ``rsa-serve``.

Subcommands
-----------
serve       start Ray, deploy the app, and (optionally) run an autoscaling
            control loop in a background thread.
autoscale   run ONLY the per-model latency autoscaler against a running
            gateway (useful as its own process/pod).
fleet       run ONLY the cross-model fleet controller -- predictive sizing,
            GPU arbitration and budget governance.
plan        print one control decision for every model and exit. The fastest
            way to answer "why did it (not) scale?" without touching the
            cluster: it reads live metrics but applies nothing.
config      print the resolved settings and exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from typing import Optional

from ray_serve_autoscale.settings import Settings, load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ray_serve_autoscale.cli")


def _http_latency_reader(base_url: str):
    """Reader that pulls the control feed from the gateway HTTP endpoint."""
    import requests

    def reader() -> dict:
        resp = requests.get(f"{base_url.rstrip('/')}/metrics/latency", timeout=5)
        resp.raise_for_status()
        return resp.json().get("latency", {})

    return reader


def _build_applier(dry_run: bool, dashboard_url: str):
    from ray_serve_autoscale.autoscaling.latency_autoscaler import (
        LoggingApplier,
        ServeRestApplier,
    )

    return LoggingApplier() if dry_run else ServeRestApplier(dashboard_url)


def _build_controller(settings: Settings, base_url: str, applier, kind: str):
    if kind == "fleet":
        from ray_serve_autoscale.autoscaling.controller import FleetAutoscaler

        return FleetAutoscaler(
            settings, reader=_http_latency_reader(base_url), applier=applier
        )
    from ray_serve_autoscale.autoscaling.latency_autoscaler import LatencyAutoscaler

    return LatencyAutoscaler(
        models=settings.models,
        cfg=settings.autoscaler,
        reader=_http_latency_reader(base_url),
        applier=applier,
    )


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
        applier = _build_applier(args.dry_run, args.dashboard_url)
        controller = _build_controller(settings, base_url, applier, args.controller)
        thread = threading.Thread(
            target=controller.run_forever, name="autoscaler", daemon=True
        )
        thread.start()
        logger.info(
            "%s autoscaler running (interval=%.0fs, dry_run=%s)",
            args.controller,
            settings.autoscaler.control_interval_s,
            args.dry_run,
        )

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
    settings = load_settings(args.config)
    applier = _build_applier(args.dry_run, args.dashboard_url)
    _build_controller(settings, args.base_url, applier, "per-model").run_forever()
    return 0


def cmd_fleet(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    applier = _build_applier(args.dry_run, args.dashboard_url)
    _build_controller(settings, args.base_url, applier, "fleet").run_forever()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """One control decision, printed and discarded. Nothing is applied."""
    from ray_serve_autoscale.autoscaling.controller import FleetAutoscaler
    from ray_serve_autoscale.autoscaling.latency_autoscaler import LoggingApplier

    settings = load_settings(args.config)
    controller = FleetAutoscaler(
        settings,
        reader=_http_latency_reader(args.base_url),
        applier=LoggingApplier(),
    )
    result = controller.step()

    if not result.healthy:
        print("controller unhealthy:")
        for note in result.notes:
            print(f"  ! {note}")
        return 1

    print(f"{'model':<16}{'now':>6}{'want':>6}{'grant':>7}{'bound':>10}  reason")
    print("-" * 100)
    for plan in result.plans:
        granted = result.granted(plan.name)
        print(
            f"{plan.name:<16}{plan.current_replicas:>6}{plan.desired_replicas:>6}"
            f"{granted if granted is not None else '-':>7}{plan.bound:>10}  {plan.reason}"
        )

    alloc = result.allocation
    if alloc is not None:
        print(
            f"\nGPU {alloc.allocated_gpus:.2f}/{alloc.usable_gpus:.2f} "
            f"({alloc.gpu_utilization:.0%})   cost ${alloc.hourly_cost_usd:.2f}/h"
            + (f" of ${alloc.budget_usd:.2f}/h" if alloc.budget_usd is not None else "")
        )
    for note in result.notes:
        print(f"  ! {note}")
    for name, why in result.skipped.items():
        print(f"  - {name}: {why}")
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
    s.add_argument("--no-autoscaler", action="store_true", help="skip the control loop")
    s.add_argument(
        "--controller",
        choices=("fleet", "per-model"),
        default="fleet",
        help="fleet: predictive + GPU arbitration; per-model: simple p95 loop",
    )
    s.add_argument("--dry-run", action="store_true", help="log decisions, don't apply")
    s.add_argument("--dashboard-url", default="http://127.0.0.1:8265")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("autoscale", help="run only the per-model latency autoscaler")
    a.add_argument("--base-url", default="http://127.0.0.1:8000")
    a.add_argument("--dashboard-url", default="http://127.0.0.1:8265")
    a.add_argument("--dry-run", action="store_true", help="log decisions, don't apply")
    a.set_defaults(func=cmd_autoscale)

    f = sub.add_parser("fleet", help="run only the cross-model fleet controller")
    f.add_argument("--base-url", default="http://127.0.0.1:8000")
    f.add_argument("--dashboard-url", default="http://127.0.0.1:8265")
    f.add_argument("--dry-run", action="store_true", help="log decisions, don't apply")
    f.set_defaults(func=cmd_fleet)

    pl = sub.add_parser("plan", help="explain one control decision and exit")
    pl.add_argument("--base-url", default="http://127.0.0.1:8000")
    pl.set_defaults(func=cmd_plan)

    c = sub.add_parser("config", help="print resolved settings and exit")
    c.set_defaults(func=cmd_config)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
