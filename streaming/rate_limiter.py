"""
SentinelAI — Custom Token-Bucket Rate Limiter.

A from-scratch implementation of the token-bucket algorithm for
rate limiting API requests. Built without external libraries so
the implementation itself is defensible in a technical interview.

Algorithm:
    - A bucket starts with `capacity` tokens
    - Tokens are added at `rate` tokens/second (refill)
    - Each request consumes 1 token
    - If the bucket is empty, the request is rejected (or waits)
    - The bucket never exceeds `capacity` tokens

Why token-bucket over leaky-bucket:
    - Token-bucket allows bursts up to `capacity` — this matches our
      traffic pattern (sustained 500 tx/sec with bursts to 2,500)
    - Leaky-bucket enforces a strict constant rate, which would reject
      legitimate burst traffic

Thread safety:
    - All state mutations are protected by threading.Lock
    - The refill calculation is atomic (single assignment after compute)

Usage:
    limiter = TokenBucketRateLimiter(rate=1000.0, capacity=2500.0)
    if limiter.acquire():
        process_request()
    else:
        return_429_too_many_requests()
"""

import threading
import time
from dataclasses import dataclass


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter.

    Attributes:
        rate: Tokens added per second (refill rate).
        capacity: Maximum tokens in the bucket (burst size).
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Tokens per second (sustained rate).
            capacity: Maximum bucket capacity (burst allowance).

        Raises:
            ValueError: If rate or capacity is non-positive.
        """
        if rate <= 0:
            raise ValueError(f"Rate must be positive, got {rate}")
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity  # Start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill.

        Must be called while holding self._lock.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.rate

        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens from the bucket (non-blocking).

        Args:
            tokens: Number of tokens to consume (default: 1).

        Returns:
            bool: True if tokens were acquired, False if bucket is empty.
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += 1
                return True
            else:
                self._total_rejected += 1
                return False

    def wait(self, tokens: int = 1, timeout: float = 5.0) -> bool:
        """Wait until tokens are available or timeout expires.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum wait time in seconds.

        Returns:
            bool: True if tokens were acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if self.acquire(tokens):
                return True

            # Calculate sleep time based on when tokens will be available
            with self._lock:
                self._refill()
                deficit = tokens - self._tokens
                if deficit <= 0:
                    continue
                wait_time = deficit / self.rate

            # Sleep for the minimum of wait_time and remaining timeout
            remaining = deadline - time.monotonic()
            sleep_time = min(wait_time, remaining, 0.1)  # Cap at 100ms
            if sleep_time <= 0:
                break
            time.sleep(sleep_time)

        return False

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens (approximate).

        Returns:
            float: Current token count after refill.
        """
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def stats(self) -> dict:
        """Rate limiter statistics.

        Returns:
            dict: Total acquired, rejected counts and current tokens.
        """
        with self._lock:
            self._refill()
            return {
                "rate": self.rate,
                "capacity": self.capacity,
                "available_tokens": round(self._tokens, 2),
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
                "rejection_rate": (
                    self._total_rejected
                    / max(
                        self._total_acquired + self._total_rejected, 1
                    )
                ),
            }
