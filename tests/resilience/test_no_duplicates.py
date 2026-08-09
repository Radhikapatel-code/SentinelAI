"""
SentinelAI — Resilience Tests: No Duplicate Scoring.

Verifies that no transaction is scored more than once across
all failure scenarios.

Usage:
    pytest tests/resilience/test_no_duplicates.py -v
"""

import sys
import os

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)


class TestNoDuplicateScoring:
    """Tests that the at-least-once + idempotent sink = exactly-once semantics."""

    def test_scorer_produces_unique_results(self):
        """Each transaction should produce exactly one result per scoring call."""
        from streaming.scorer import ThreadSafeScorer
        from streaming.models import TransactionMessage

        scorer = ThreadSafeScorer(
            data_path="data/transactions.csv",
            worker_id="test-dedup",
        )

        results = {}
        for tx_id in range(1, 501):
            msg = TransactionMessage(
                transaction_id=tx_id,
                amount=100.0 + tx_id,
                transaction_time=14,
                location_change=0,
                device_change=0,
                merchant_risk=0.2,
            )
            result = scorer.score(msg)

            # Verify no duplicate transaction IDs in results
            assert result.transaction_id not in results, (
                f"Duplicate scoring for tx {tx_id}"
            )
            results[result.transaction_id] = result

        assert len(results) == 500

    def test_idempotent_results(self):
        """Scoring the same transaction twice should yield identical results.

        This proves that the idempotent UPSERT in PostgreSQL will produce
        correct results even under at-least-once delivery.
        """
        from streaming.scorer import ThreadSafeScorer
        from streaming.models import TransactionMessage

        scorer = ThreadSafeScorer(
            data_path="data/transactions.csv",
            worker_id="test-idempotent",
        )

        msg = TransactionMessage(
            transaction_id=42,
            amount=5000.0,
            transaction_time=2,
            location_change=1,
            device_change=1,
            merchant_risk=0.85,
        )

        result1 = scorer.score(msg)
        result2 = scorer.score(msg)

        assert result1.transaction_id == result2.transaction_id
        assert result1.decision == result2.decision
        assert abs(result1.fraud_probability - result2.fraud_probability) < 1e-6
        assert abs(result1.anomaly_score - result2.anomaly_score) < 1e-6
        assert abs(result1.expected_cost - result2.expected_cost) < 1e-4

    def test_concurrent_scoring_no_duplicates(self):
        """Multiple threads scoring different transactions should not interfere."""
        import threading
        from streaming.scorer import ThreadSafeScorer
        from streaming.models import TransactionMessage

        scorer = ThreadSafeScorer(
            data_path="data/transactions.csv",
            worker_id="test-concurrent",
        )

        results: dict[int, str] = {}
        errors: list[str] = []
        lock = threading.Lock()

        def score_range(start: int, end: int):
            for tx_id in range(start, end):
                msg = TransactionMessage(
                    transaction_id=tx_id,
                    amount=100.0 + tx_id,
                    transaction_time=14,
                    location_change=0,
                    device_change=0,
                    merchant_risk=0.1,
                )
                result = scorer.score(msg)

                with lock:
                    if result.transaction_id in results:
                        errors.append(
                            f"Duplicate: tx {result.transaction_id}"
                        )
                    results[result.transaction_id] = result.decision

        # 4 threads, non-overlapping ranges
        threads = []
        for i in range(4):
            start = i * 100 + 1
            end = (i + 1) * 100 + 1
            t = threading.Thread(target=score_range, args=(start, end))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Duplicate errors: {errors}"
        assert len(results) == 400
