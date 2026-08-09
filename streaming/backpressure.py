"""
SentinelAI — Backpressure Monitoring.

Monitors Kafka consumer lag and signals when the system is overloaded.
When lag exceeds a threshold, the system surfaces a degraded-mode signal
via Redis and the API health endpoint.

Backpressure strategy:
    1. Monitor consumer lag via confluent-kafka's consumer statistics
    2. When lag > threshold → set Redis key "sentinel:backpressure" = "active"
    3. Producer/ingestion API checks this key and throttles
    4. Health endpoint reports degraded status
    5. Prometheus gauge tracks lag for Grafana alerting

This is a cooperative backpressure mechanism, not a hard stop.
The producer voluntarily throttles; it isn't blocked.
"""

import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class BackpressureMonitor:
    """Monitors consumer lag and signals backpressure.

    Attributes:
        lag_threshold: Consumer lag that triggers backpressure.
        check_interval: Seconds between lag checks.
        is_active: Whether backpressure is currently signaled.
    """

    def __init__(
        self,
        lag_threshold: int = 5000,
        check_interval: float = 5.0,
        redis_url: Optional[str] = None,
    ) -> None:
        """Initialize the backpressure monitor.

        Args:
            lag_threshold: Lag count that triggers backpressure.
            check_interval: Seconds between checks.
            redis_url: Redis URL for signaling (optional).
        """
        self.lag_threshold = lag_threshold
        self.check_interval = check_interval
        self._is_active = False
        self._current_lag = 0
        self._lock = threading.Lock()
        self._redis = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Metrics
        self._activations = 0
        self._deactivations = 0
        self._max_lag_seen = 0

        if redis_url:
            try:
                import redis as redis_lib
                self._redis = redis_lib.from_url(
                    redis_url, decode_responses=True
                )
                self._redis.ping()
            except Exception:
                logger.warning(
                    "Redis unavailable for backpressure signaling",
                    exc_info=True,
                )
                self._redis = None

    @property
    def is_active(self) -> bool:
        """Whether backpressure is currently active."""
        with self._lock:
            return self._is_active

    @property
    def current_lag(self) -> int:
        """Current consumer lag."""
        with self._lock:
            return self._current_lag

    def update_lag(self, lag: int) -> None:
        """Update the current consumer lag and check threshold.

        Called by the consumer worker after each poll batch.

        Args:
            lag: Current consumer lag (number of unprocessed messages).
        """
        with self._lock:
            self._current_lag = lag
            self._max_lag_seen = max(self._max_lag_seen, lag)

            was_active = self._is_active

            if lag >= self.lag_threshold and not self._is_active:
                self._is_active = True
                self._activations += 1
                logger.warning(
                    "⚠️ BACKPRESSURE ACTIVE: lag=%d (threshold=%d)",
                    lag, self.lag_threshold,
                )
            elif lag < self.lag_threshold * 0.8 and self._is_active:
                # Hysteresis: deactivate at 80% of threshold to avoid flapping
                self._is_active = False
                self._deactivations += 1
                logger.info(
                    "✅ BACKPRESSURE CLEARED: lag=%d", lag,
                )

        # Signal via Redis (outside lock)
        if self._redis:
            try:
                if self._is_active:
                    self._redis.set(
                        "sentinel:backpressure",
                        json.dumps({
                            "active": True,
                            "lag": lag,
                            "threshold": self.lag_threshold,
                            "timestamp": time.time(),
                        }),
                        ex=60,  # Auto-expire in 60s as safety net
                    )
                elif was_active and not self._is_active:
                    self._redis.delete("sentinel:backpressure")
            except Exception:
                logger.debug("Redis backpressure signal failed", exc_info=True)

    def should_throttle(self) -> bool:
        """Check if the producer should throttle.

        Used by the producer/ingestion API to cooperatively reduce rate.

        Returns:
            bool: True if backpressure is active.
        """
        # Check local state first (fast path)
        if self._is_active:
            return True

        # Check Redis signal (for cross-process signaling)
        if self._redis:
            try:
                val = self._redis.get("sentinel:backpressure")
                if val:
                    data = json.loads(val)
                    return data.get("active", False)
            except Exception:
                pass

        return False

    @property
    def stats(self) -> dict:
        """Backpressure monitor statistics.

        Returns:
            dict: Current state and historical metrics.
        """
        with self._lock:
            return {
                "is_active": self._is_active,
                "current_lag": self._current_lag,
                "lag_threshold": self.lag_threshold,
                "max_lag_seen": self._max_lag_seen,
                "activations": self._activations,
                "deactivations": self._deactivations,
            }
