"""
SentinelAI — Resilience Tests: Slow Scoring (Circuit Breaker).

Tests that the circuit breaker correctly trips when scoring is degraded:
    1. Circuit opens after threshold consecutive failures
    2. CircuitBreakerOpenError is raised when circuit is open
    3. Circuit transitions to HALF_OPEN after recovery timeout
    4. Circuit closes after successful recovery

Usage:
    pytest tests/resilience/test_slow_scoring.py -v
"""

import time
import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from streaming.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


class TestCircuitBreakerResilience:
    """Tests for circuit breaker behavior under degraded scoring."""

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit should open after failure_threshold consecutive failures."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2,
        )

        def failing_func():
            raise RuntimeError("Scoring timeout")

        # Trigger failures up to threshold
        for i in range(3):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

    def test_circuit_rejects_when_open(self):
        """Calls should be immediately rejected when circuit is open."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=10.0,
        )

        def failing_func():
            raise RuntimeError("fail")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)

        # Next call should be rejected without executing the function
        call_count = 0

        def tracked_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(tracked_func)

        assert call_count == 0, "Function should not have been called"

    def test_circuit_transitions_to_half_open(self):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.5,  # 500ms for fast test
            success_threshold=2,
        )

        def failing_func():
            raise RuntimeError("fail")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.6)

        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_recovery(self):
        """Circuit should close after success_threshold successes in HALF_OPEN."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.3,
            success_threshold=2,
        )

        def failing_func():
            raise RuntimeError("fail")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)

        # Wait for HALF_OPEN
        time.sleep(0.4)
        assert cb.state == CircuitState.HALF_OPEN

        # Successful calls should close the circuit
        def success_func():
            return "ok"

        cb.call(success_func)
        cb.call(success_func)

        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Any failure in HALF_OPEN should reopen the circuit."""
        cb = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.3,
            success_threshold=3,
        )

        def failing_func():
            raise RuntimeError("fail")

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)

        # Wait for HALF_OPEN
        time.sleep(0.4)
        assert cb.state == CircuitState.HALF_OPEN

        # One success, then a failure
        cb.call(lambda: "ok")

        with pytest.raises(RuntimeError):
            cb.call(failing_func)

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_stats(self):
        """Stats should accurately reflect circuit breaker activity."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
        )

        # 2 successes
        cb.call(lambda: "ok")
        cb.call(lambda: "ok")

        # 3 failures → trip
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        # 1 rejection
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "ok")

        stats = cb.stats
        assert stats["total_calls"] == 6  # 2 + 3 + 1
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 3
        assert stats["total_rejected"] == 1
        assert stats["state"] == "open"

    def test_slow_scoring_simulation(self):
        """Simulate slow scoring and verify circuit breaker trips."""
        cb = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.5,
            success_threshold=2,
        )

        call_latencies = []

        def slow_scorer():
            """Simulates a scorer that times out."""
            time.sleep(0.1)  # Simulate slow scoring
            raise TimeoutError("Scoring took too long")

        def fast_scorer():
            """Normal-speed scorer."""
            return {"decision": "approve"}

        # Phase 1: Slow scoring triggers circuit breaker
        for _ in range(3):
            start = time.time()
            try:
                cb.call(slow_scorer)
            except (TimeoutError, CircuitBreakerOpenError):
                pass
            call_latencies.append(time.time() - start)

        assert cb.state == CircuitState.OPEN

        # Phase 2: Subsequent calls are rejected immediately (fast fail)
        start = time.time()
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(fast_scorer)
        rejection_time = time.time() - start

        # Rejection should be near-instant (< 10ms)
        assert rejection_time < 0.01, (
            f"Circuit breaker rejection took {rejection_time:.3f}s, "
            "should be near-instant"
        )

        # Phase 3: After recovery, circuit closes with fast scoring
        time.sleep(0.6)
        assert cb.state == CircuitState.HALF_OPEN

        cb.call(fast_scorer)
        cb.call(fast_scorer)
        assert cb.state == CircuitState.CLOSED
