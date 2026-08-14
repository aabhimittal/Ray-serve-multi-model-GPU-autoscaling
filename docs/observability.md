# Observability

The gateway exposes Prometheus exposition at `GET /metrics`. Import
[`deploy/grafana/dashboard.json`](../deploy/grafana/dashboard.json) for a
dashboard laid out around the four questions an operator actually asks during an
incident.

## The series that matter

| Series | Question it answers |
|--------|--------------------|
| `rsa_slo_attainment_ratio` | **Is the SLO being met?** p95 ÷ SLO. Normalised across models, so one threshold works for all of them. |
| `rsa_queue_p95_ms` / `rsa_compute_p95_ms` | **Why is it slow?** Queue-dominated means scaling out helps; compute-dominated means it won't. |
| `rsa_replicas_ready` / `rsa_replicas_pending` | **Is capacity arriving?** Pending replicas are paid for but not yet serving. |
| `rsa_replicas_unmet` | **Is the cluster the constraint?** Demand arbitration could not grant. |
| `rsa_shed_total` / `rsa_breaker_state` | **Are we shedding, and why?** |
| `rsa_hourly_cost_usd` | **What is this costing?** |

Also exported: `rsa_requests_total`, `rsa_errors_total`, `rsa_inflight_requests`,
`rsa_latency_p50/p95/p99_ms`, `rsa_requests_per_second`, `rsa_slo_ms`,
`rsa_gpus_allocated`, `rsa_admitted_total`.

## Alerting

`rsa_slo_attainment_ratio` is the one series worth paging on. Alert on *sustained*
breach, not instantaneous — the controller is already given a confirmation window
before it acts, and an alert that fires faster than the system can respond only
trains people to ignore it.

```yaml
groups:
  - name: ray-serve-autoscale
    rules:
      - alert: ModelMissingLatencySLO
        expr: rsa_slo_attainment_ratio > 1
        for: 5m
        labels: { severity: page }
        annotations:
          summary: "{{ $labels.model }} is missing its latency SLO"
          description: "p95 is {{ $value | humanize }}x the SLO. Check rsa_queue_p95_ms vs rsa_compute_p95_ms: queue-dominated means capacity, compute-dominated means the model itself."

      # Sustained unmet demand means the *cluster* is the binding constraint.
      # No amount of policy tuning fixes it -- add GPUs or lower max_replicas.
      - alert: FleetGpuContention
        expr: sum(rsa_replicas_unmet) > 0
        for: 15m
        labels: { severity: ticket }
        annotations:
          summary: "Autoscaler demand exceeds cluster GPU capacity"

      # Shedding during a ramp is the system working. Shedding for 10 minutes
      # is the system failing to catch up.
      - alert: SustainedLoadShedding
        expr: rate(rsa_shed_total[5m]) > 1
        for: 10m
        labels: { severity: page }
        annotations:
          summary: "{{ $labels.model }} has been shedding load for 10m"

      - alert: CircuitBreakerOpen
        expr: rsa_breaker_state == 2
        for: 2m
        labels: { severity: page }
        annotations:
          summary: "{{ $labels.model }} circuit breaker is open (backend failing)"

      # A frozen exporter is the most dangerous failure mode for a closed loop:
      # the controller holds while the fleet degrades. The controller detects
      # this itself and refuses to act, but a human should still know.
      - alert: MetricsFeedStale
        expr: rate(rsa_requests_total[10m]) == 0 and rsa_inflight_requests > 0
        for: 10m
        labels: { severity: ticket }
        annotations:
          summary: "{{ $labels.model }} metrics feed may be stale"
```

## Debugging a scaling decision

Prometheus tells you *what* happened; the controller tells you *why*:

```bash
rsa-serve --config config/models.yaml plan
```

It reads live metrics, prints one line of reasoning per model, and applies
nothing. Every plan carries a human-readable justification by construction — an
autoscaler that cannot explain itself cannot be operated.

For the control loop's own view over time, run it in dry-run mode and watch the
log: `rsa-serve fleet --dry-run`.

## Ray's own metrics

Ray Serve exports its own Prometheus metrics (`ray_serve_*`) covering replica
counts, queued requests and the built-in autoscaler's decisions. They complement
these: `ray_serve_*` describes the *mechanism*, `rsa_*` describes the *policy*
and the SLO. Scrape both.
