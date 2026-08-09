"""
SentinelAI — Scorer Thread-Safety Tests.

Verifies that the ThreadSafeScorer produces correct, deterministic
results under concurrent access from multiple threads.

This is the evidence that the scoring pipeline is safe for use
in multi-threaded/multi-process workers.

Usage:
    pytest tests/test_scorer_thread_safety.py -v
"""

import os
import sys
import threading

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from streaming.scorer import ThreadSafeScorer
from streaming.models import TransactionMessage


@pytest.fixture(scope="module")
def scorer():
    """Shared scorer instance across all tests in this module.

    Using a single instance across threads is the exact scenario
    we need to verify is safe.
    """
    return ThreadSafeScorer(
        data_path="data/transactions.csv",
        worker_id="thread-safety-test",
    )


class TestScorerThreadSafety:
    """Tests for concurrent scoring correctness."""

    def test_single_thread_deterministic(self, scorer):
        """Same input should always produce the same output."""
        msg = TransactionMessage(
            transaction_id=1,
            amount=5000.0,
            transaction_time=2,
            location_change=1,
            device_change=1,
            merchant_risk=0.85,
        )

        results = [scorer.score(msg) for _ in range(10)]

        # All results should be identical
        for r in results[1:]:
            assert r.decision == results[0].decision
            assert abs(r.fraud_probability - results[0].fraud_probability) < 1e-10
            assert abs(r.anomaly_score - results[0].anomaly_score) < 1e-10

    def test_concurrent_scoring_correctness(self, scorer):
        """100 transactions scored across 4 threads should all be correct.

        Strategy:
            1. Score all 100 transactions single-threaded (baseline)
            2. Score all 100 transactions across 4 threads
            3. Compare results — they must match exactly
        """
        # Generate test messages
        messages = []
        for i in range(100):
            messages.append(TransactionMessage(
                transaction_id=i + 1,
                amount=100.0 + (i * 50),
                transaction_time=i % 24,
                location_change=i % 3 == 0,
                device_change=i % 7 == 0,
                merchant_risk=round((i % 100) / 100, 2),
            ))

        # Baseline: single-threaded
        baseline = {}
        for msg in messages:
            result = scorer.score(msg)
            baseline[msg.transaction_id] = (
                result.decision,
                round(result.fraud_probability, 10),
                round(result.anomaly_score, 10),
            )

        # Multi-threaded
        concurrent_results = {}
        errors = []
        lock = threading.Lock()

        def score_chunk(chunk):
            for msg in chunk:
                try:
                    result = scorer.score(msg)
                    with lock:
                        concurrent_results[msg.transaction_id] = (
                            result.decision,
                            round(result.fraud_probability, 10),
                            round(result.anomaly_score, 10),
                        )
                except Exception as e:
                    with lock:
                        errors.append((msg.transaction_id, str(e)))

        # Split messages across 4 threads
        chunk_size = 25
        threads = []
        for i in range(4):
            chunk = messages[i * chunk_size:(i + 1) * chunk_size]
            t = threading.Thread(target=score_chunk, args=(chunk,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify
        assert len(errors) == 0, f"Scoring errors: {errors}"
        assert len(concurrent_results) == 100

        for tx_id in baseline:
            assert tx_id in concurrent_results, f"Missing tx {tx_id}"
            assert baseline[tx_id] == concurrent_results[tx_id], (
                f"Mismatch for tx {tx_id}: "
                f"baseline={baseline[tx_id]}, "
                f"concurrent={concurrent_results[tx_id]}"
            )

    def test_high_contention(self, scorer):
        """All threads scoring the SAME transaction simultaneously.

        This is the worst case for contention — all threads hit
        the same sklearn predict path at the same time.
        """
        msg = TransactionMessage(
            transaction_id=999,
            amount=89000.0,
            transaction_time=2,
            location_change=1,
            device_change=1,
            merchant_risk=0.95,
        )

        results = []
        errors = []
        lock = threading.Lock()

        def score_same():
            try:
                result = scorer.score(msg)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=score_same) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 8

        # All results should be identical
        for r in results[1:]:
            assert r.decision == results[0].decision
            assert abs(r.fraud_probability - results[0].fraud_probability) < 1e-10

    def test_scorer_produces_valid_results(self, scorer):
        """Verify result fields are populated correctly."""
        msg = TransactionMessage(
            transaction_id=42,
            amount=1200.0,
            transaction_time=14,
            location_change=0,
            device_change=0,
            merchant_risk=0.05,
        )

        result = scorer.score(msg)

        assert result.transaction_id == 42
        assert result.amount == 1200.0
        assert 0.0 <= result.fraud_probability <= 1.0
        assert result.decision in ("approve", "block", "review")
        assert result.expected_cost >= 0
        assert result.scoring_latency_ms > 0
        assert result.worker_id == "thread-safety-test"

    def test_different_inputs_different_outputs(self, scorer):
        """Distinct transactions should produce different results.

        Verifies the scorer isn't accidentally caching or reusing
        results across calls.
        """
        low_risk = TransactionMessage(
            transaction_id=1,
            amount=100.0,
            transaction_time=14,
            location_change=0,
            device_change=0,
            merchant_risk=0.01,
        )

        high_risk = TransactionMessage(
            transaction_id=2,
            amount=89000.0,
            transaction_time=2,
            location_change=1,
            device_change=1,
            merchant_risk=0.99,
        )

        r_low = scorer.score(low_risk)
        r_high = scorer.score(high_risk)

        # High-risk should have higher fraud probability
        assert r_high.fraud_probability > r_low.fraud_probability
        # Decisions should differ (low → approve, high → block likely)
        # (Not guaranteed depending on model, so just check they're valid)
        assert r_low.decision in ("approve", "block", "review")
        assert r_high.decision in ("approve", "block", "review")
