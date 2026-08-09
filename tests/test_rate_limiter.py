"""
SentinelAI — Token-Bucket Rate Limiter Unit Tests.

Tests correctness of the custom rate limiter implementation:
    1. Basic acquire/reject behavior
    2. Token refill over time
    3. Burst capacity
    4. Thread safety under concurrent access
    5. Wait (blocking) mode
    6. Edge cases and stats tracking

Usage:
    pytest tests/test_rate_limiter.py -v
"""

import sys
import os
import threading
import time

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from streaming.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Unit tests for the custom token-bucket rate limiter."""

    def test_initial_capacity_is_full(self):
        """Bucket should start at full capacity."""
        limiter = TokenBucketRateLimiter(rate=100, capacity=50)
        assert limiter.available_tokens == pytest.approx(50, abs=1)

    def test_acquire_consumes_token(self):
        """Each acquire should consume one token."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=5)

        for _ in range(5):
            assert limiter.acquire() is True

        # Bucket should be empty now (approximately)
        assert limiter.available_tokens < 1

    def test_reject_when_empty(self):
        """Acquire should return False when bucket is empty."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=3)

        # Drain the bucket
        for _ in range(3):
            limiter.acquire()

        # Next acquire should fail
        assert limiter.acquire() is False

    def test_refill_over_time(self):
        """Tokens should refill based on elapsed time."""
        limiter = TokenBucketRateLimiter(rate=100, capacity=100)

        # Drain all tokens
        for _ in range(100):
            limiter.acquire()

        # Wait for refill (100 tokens/sec → 10 tokens in 100ms)
        time.sleep(0.15)

        # Should have refilled some tokens
        assert limiter.available_tokens > 5
        assert limiter.acquire() is True

    def test_capacity_is_maximum(self):
        """Tokens should never exceed capacity, even with long wait."""
        limiter = TokenBucketRateLimiter(rate=1000, capacity=50)

        # Wait for potential over-refill
        time.sleep(0.1)

        assert limiter.available_tokens <= 50

    def test_burst_allowance(self):
        """Full bucket should allow burst up to capacity."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=100)

        # Burst: acquire 100 tokens rapidly
        acquired = sum(1 for _ in range(100) if limiter.acquire())
        assert acquired == 100

        # Post-burst: should be rate-limited
        assert limiter.acquire() is False

    def test_wait_mode_blocks_until_available(self):
        """Wait mode should block and eventually succeed."""
        limiter = TokenBucketRateLimiter(rate=100, capacity=5)

        # Drain the bucket
        for _ in range(5):
            limiter.acquire()

        # Wait should succeed after refill
        start = time.monotonic()
        result = limiter.wait(tokens=1, timeout=1.0)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed < 0.5  # Should refill within ~10ms at 100/sec

    def test_wait_mode_timeout(self):
        """Wait should return False after timeout if rate is too slow."""
        limiter = TokenBucketRateLimiter(rate=1, capacity=1)

        # Drain the bucket
        limiter.acquire()

        # Request 5 tokens with short timeout (would need 5 seconds at rate=1)
        result = limiter.wait(tokens=5, timeout=0.2)
        assert result is False

    def test_thread_safety(self):
        """Concurrent access should not corrupt internal state."""
        limiter = TokenBucketRateLimiter(rate=10000, capacity=5000)

        acquired_count = [0]
        rejected_count = [0]
        lock = threading.Lock()

        def hammer(iterations):
            local_acquired = 0
            local_rejected = 0
            for _ in range(iterations):
                if limiter.acquire():
                    local_acquired += 1
                else:
                    local_rejected += 1
            with lock:
                acquired_count[0] += local_acquired
                rejected_count[0] += local_rejected

        # 8 threads, each trying 1000 acquires
        threads = []
        for _ in range(8):
            t = threading.Thread(target=hammer, args=(1000,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        total_attempts = acquired_count[0] + rejected_count[0]
        assert total_attempts == 8000

        # Tokens should never go negative
        assert limiter.available_tokens >= 0

        # Should have acquired at most capacity + refilled tokens
        assert acquired_count[0] <= 5000 + 10000  # capacity + rate*time

    def test_stats_tracking(self):
        """Stats should accurately reflect limiter activity."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=3)

        limiter.acquire()  # success
        limiter.acquire()  # success
        limiter.acquire()  # success
        limiter.acquire()  # reject (bucket empty)

        stats = limiter.stats
        assert stats["rate"] == 10
        assert stats["capacity"] == 3
        assert stats["total_acquired"] == 3
        assert stats["total_rejected"] == 1
        assert stats["rejection_rate"] == pytest.approx(0.25, abs=0.01)

    def test_invalid_rate_raises(self):
        """Non-positive rate should raise ValueError."""
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(rate=0, capacity=10)

        with pytest.raises(ValueError):
            TokenBucketRateLimiter(rate=-5, capacity=10)

    def test_invalid_capacity_raises(self):
        """Non-positive capacity should raise ValueError."""
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(rate=10, capacity=0)

    def test_multiple_token_acquire(self):
        """Acquiring multiple tokens at once should work correctly."""
        limiter = TokenBucketRateLimiter(rate=10, capacity=10)

        # Acquire 5 tokens at once
        assert limiter.acquire(tokens=5) is True
        assert limiter.available_tokens == pytest.approx(5, abs=1)

        # Acquire another 5
        assert limiter.acquire(tokens=5) is True

        # Should be empty now
        assert limiter.acquire(tokens=5) is False
