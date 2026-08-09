"""
SentinelAI — Prometheus Metrics Instrumentation.

Instruments the scoring pipeline with Prometheus metrics for
real-time observability via Grafana.

Key metrics:
    - sentinel_transactions_scored_total: Counter (by worker, decision)
    - sentinel_scoring_latency_seconds: Histogram (p50/p95/p99)
    - sentinel_queue_depth: Gauge (current consumer lag)
    - sentinel_error_total: Counter (by error type)
    - sentinel_circuit_breaker_state: Gauge (0=closed, 1=open, 2=half_open)
    - sentinel_false_negative_rate: Gauge (the headline metric)
    - sentinel_rate_limiter_rejections_total: Counter

Usage:
    from streaming.metrics import metrics, start_metrics_server
    start_metrics_server(port=8001)

    metrics.record_scoring(worker_id="w1", decision="block", latency=0.003)
"""

import logging
from typing import Optional

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    start_http_server,
)

logger = logging.getLogger(__name__)


class SentinelMetrics:
    """Centralized Prometheus metrics for SentinelAI.

    All metrics use the 'sentinel_' prefix for namespacing.
    """

    def __init__(self) -> None:
        """Initialize all Prometheus metrics."""

        # ── Throughput ──
        self.transactions_scored = Counter(
            "sentinel_transactions_scored_total",
            "Total transactions scored",
            labelnames=["worker_id", "decision"],
        )

        self.transactions_ingested = Counter(
            "sentinel_transactions_ingested_total",
            "Total transactions ingested by the producer",
        )

        # ── Latency ──
        self.scoring_latency = Histogram(
            "sentinel_scoring_latency_seconds",
            "Time to score a single transaction",
            labelnames=["worker_id"],
            buckets=(
                0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
            ),
        )

        self.end_to_end_latency = Histogram(
            "sentinel_end_to_end_latency_seconds",
            "Time from ingestion to sink write",
            buckets=(
                0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0,
            ),
        )

        # ── Queue ──
        self.queue_depth = Gauge(
            "sentinel_queue_depth",
            "Current consumer lag (unprocessed messages)",
            labelnames=["worker_id"],
        )

        # ── Errors ──
        self.errors = Counter(
            "sentinel_error_total",
            "Total errors by type",
            labelnames=["error_type"],
        )

        # ── Circuit Breaker ──
        # 0 = closed, 1 = open, 2 = half_open
        self.circuit_breaker_state = Gauge(
            "sentinel_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
        )

        # ── Rate Limiter ──
        self.rate_limiter_rejections = Counter(
            "sentinel_rate_limiter_rejections_total",
            "Requests rejected by rate limiter",
        )

        # ── Model Quality ──
        self.false_negative_rate = Gauge(
            "sentinel_false_negative_rate",
            "Fraction of fraud transactions approved (false negatives)",
        )

        self.false_positive_rate = Gauge(
            "sentinel_false_positive_rate",
            "Fraction of legitimate transactions blocked (false positives)",
        )

        # ── Backpressure ──
        self.backpressure_active = Gauge(
            "sentinel_backpressure_active",
            "Whether backpressure is currently active (0 or 1)",
        )

        # ── Worker Info ──
        self.worker_info = Info(
            "sentinel_worker",
            "Worker metadata",
        )

        # ── Running counts for FNR calculation ──
        self._fraud_approved = 0
        self._fraud_total = 0
        self._legit_blocked = 0
        self._legit_total = 0

    def record_scoring(
        self,
        worker_id: str,
        decision: str,
        latency_seconds: float,
        is_fraud: Optional[bool] = None,
        ingested_at: Optional[float] = None,
        scored_at: Optional[float] = None,
    ) -> None:
        """Record a completed scoring event.

        Args:
            worker_id: Worker that performed the scoring.
            decision: Action chosen ("approve", "block", "review").
            latency_seconds: Scoring latency in seconds.
            is_fraud: Ground-truth label (for FNR tracking).
            ingested_at: When the transaction was ingested (epoch).
            scored_at: When scoring completed (epoch).
        """
        self.transactions_scored.labels(
            worker_id=worker_id, decision=decision
        ).inc()

        self.scoring_latency.labels(worker_id=worker_id).observe(
            latency_seconds
        )

        # End-to-end latency
        if ingested_at and scored_at:
            e2e = scored_at - ingested_at
            if e2e > 0:
                self.end_to_end_latency.observe(e2e)

        # False negative rate tracking
        if is_fraud is not None:
            if is_fraud:
                self._fraud_total += 1
                if decision == "approve":
                    self._fraud_approved += 1
            else:
                self._legit_total += 1
                if decision == "block":
                    self._legit_blocked += 1

            # Update gauges
            if self._fraud_total > 0:
                self.false_negative_rate.set(
                    self._fraud_approved / self._fraud_total
                )
            if self._legit_total > 0:
                self.false_positive_rate.set(
                    self._legit_blocked / self._legit_total
                )

    def record_error(self, error_type: str) -> None:
        """Record an error event.

        Args:
            error_type: Category of error (e.g., "deserialization",
                "scoring", "sink_write").
        """
        self.errors.labels(error_type=error_type).inc()


# Singleton metrics instance
metrics = SentinelMetrics()


def start_metrics_server(port: int = 8001) -> None:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to expose metrics on.
    """
    start_http_server(port)
    logger.info("Prometheus metrics server started on port %d", port)
