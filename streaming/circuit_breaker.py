"""
SentinelAI — Circuit Breaker Pattern.

Protects the scoring call path from cascading failures when
downstream scoring is degraded or slow. Instead of queueing
indefinitely, the circuit breaker fails fast and sheds load.

State machine:
    CLOSED  → Normal operation. Requests pass through.
              If `failure_threshold` consecutive failures occur,
              transitions to OPEN.

    OPEN    → Failing fast. All requests immediately return a
              fallback response without attempting scoring.
              After `recovery_timeout` seconds, transitions to HALF_OPEN.

    HALF_OPEN → Testing recovery. A limited number of requests
                are allowed through. If `success_threshold` consecutive
                successes occur, transitions to CLOSED.
                Any failure transitions back to OPEN.

Usage:
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

    try:
        result = cb.call(scorer.score, transaction)
    except CircuitBreakerOpenError:
        # System is degraded — return fallback response
        return degraded_response(transaction)
"""

import enum
import logging
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""

    def __init__(self, recovery_time: float) -> None:
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker is OPEN. "
            f"Recovery in {recovery_time:.1f}s."
        )


class CircuitBreaker:
    """Thread-safe circuit breaker for the scoring call path.

    Attributes:
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds before testing recovery.
        success_threshold: Successes in HALF_OPEN before closing.
        state: Current circuit state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        name: str = "scoring",
    ) -> None:
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Failures before tripping to OPEN.
            recovery_timeout: Seconds to wait before HALF_OPEN test.
            success_threshold: Successes needed to close from HALF_OPEN.
            name: Name for logging and metrics.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

        # Metrics
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejected = 0
        self._state_changes: list[tuple[float, str, str]] = []

    @property
    def state(self) -> CircuitState:
        """Current circuit breaker state."""
        with self._lock:
            self._check_recovery()
            return self._state

    def _check_recovery(self) -> None:
        """Check if OPEN circuit should transition to HALF_OPEN.

        Must be called while holding self._lock.
        """
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self.recovery_timeout:
                self._transition(CircuitState.HALF_OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging.

        Must be called while holding self._lock.

        Args:
            new_state: Target state.
        """
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        self._state_changes.append(
            (time.time(), old_state.value, new_state.value)
        )

        logger.warning(
            "Circuit breaker '%s': %s → %s",
            self.name, old_state.value, new_state.value,
        )

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the circuit breaker.

        Args:
            func: The function to call (e.g., scorer.score).
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The function's return value.

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN.
            Exception: Any exception from the wrapped function
                (after recording the failure).
        """
        with self._lock:
            self._total_calls += 1
            self._check_recovery()

            if self._state == CircuitState.OPEN:
                self._total_rejected += 1
                remaining = self.recovery_timeout - (
                    time.monotonic() - self._opened_at
                )
                raise CircuitBreakerOpenError(max(remaining, 0))

        # Execute the function (outside the lock)
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._total_successes += 1
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN → back to OPEN
                self._transition(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition(CircuitState.OPEN)

    @property
    def stats(self) -> dict:
        """Circuit breaker statistics.

        Returns:
            dict: State, call counts, and recent state transitions.
        """
        with self._lock:
            self._check_recovery()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "total_successes": self._total_successes,
                "total_failures": self._total_failures,
                "total_rejected": self._total_rejected,
                "recent_transitions": self._state_changes[-10:],
            }

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._transition(CircuitState.CLOSED)
