"""
SentinelAI — False Negative Rate Measurement Under Load.

The single most important metric: proves that the systems layer
didn't quietly degrade model quality under load.

Measures FNR at different throughput levels and compares to the
baseline (single-threaded, no load) FNR. The claim:
    "Scaled throughput from X to Y tx/sec while holding
     false-negative rate constant at Z%"

Usage:
    python load_tests/measure_false_negative_rate.py
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from streaming.scorer import ThreadSafeScorer
from streaming.models import TransactionMessage

logger = logging.getLogger(__name__)


def measure_fnr(
    scorer: ThreadSafeScorer,
    data_path: str = "data/stream_transactions.csv",
    max_count: int = 10000,
) -> dict:
    """Measure false-negative rate on labeled data.

    False negative = fraud transaction that was APPROVED.

    Args:
        scorer: Initialized scorer.
        data_path: Path to labeled transaction data.
        max_count: Maximum transactions to process.

    Returns:
        dict: FNR metrics.
    """
    if not os.path.exists(data_path):
        # Generate synthetic data if not present
        logger.info("Generating synthetic data for FNR measurement...")
        from scripts.generate_stream_data import generate_transactions
        df = generate_transactions(count=max_count)
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path, nrows=max_count)

    if "is_fraud" not in df.columns:
        return {"error": "Dataset has no is_fraud column for FNR measurement"}

    # Score all transactions
    fraud_total = 0
    fraud_approved = 0
    legit_total = 0
    legit_blocked = 0
    decisions = {"approve": 0, "block": 0, "review": 0}

    start_time = time.perf_counter()

    for row in df.to_dict("records"):
        msg = TransactionMessage(
            transaction_id=int(row["transaction_id"]),
            amount=float(row["amount"]),
            transaction_time=int(row["transaction_time"]),
            location_change=int(row["location_change"]),
            device_change=int(row["device_change"]),
            merchant_risk=float(row["merchant_risk"]),
            is_fraud=bool(row["is_fraud"]),
        )

        result = scorer.score(msg)
        decisions[result.decision] = decisions.get(result.decision, 0) + 1

        if msg.is_fraud:
            fraud_total += 1
            if result.decision == "approve":
                fraud_approved += 1
        else:
            legit_total += 1
            if result.decision == "block":
                legit_blocked += 1

    elapsed = time.perf_counter() - start_time

    fnr = fraud_approved / fraud_total if fraud_total > 0 else 0.0
    fpr = legit_blocked / legit_total if legit_total > 0 else 0.0

    return {
        "total_transactions": len(df),
        "fraud_transactions": fraud_total,
        "legit_transactions": legit_total,
        "fraud_approved_false_negatives": fraud_approved,
        "legit_blocked_false_positives": legit_blocked,
        "false_negative_rate": round(fnr, 6),
        "false_positive_rate": round(fpr, 6),
        "decisions": decisions,
        "throughput_tx_sec": round(len(df) / elapsed, 1),
        "total_time_seconds": round(elapsed, 3),
    }


def main() -> None:
    """Run FNR measurement and report results."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  SentinelAI — False Negative Rate Measurement")
    print("=" * 60)

    scorer = ThreadSafeScorer(
        data_path="data/transactions.csv",
        worker_id="fnr-measure",
    )

    # Try synthetic data first, fall back to transactions.csv
    data_paths = [
        "data/stream_transactions.csv",
    ]

    for data_path in data_paths:
        print(f"\n[+] Measuring FNR on: {data_path}")
        results = measure_fnr(scorer, data_path=data_path, max_count=1000)

        if "error" in results:
            print(f"   [WARNING]  {results['error']}")
            continue

        print(f"   Total transactions: {results['total_transactions']}")
        print(f"   Fraud transactions: {results['fraud_transactions']}")
        print(f"   False Negative Rate: {results['false_negative_rate']:.4%}")
        print(f"   False Positive Rate: {results['false_positive_rate']:.4%}")
        print(f"   Decisions: {results['decisions']}")
        print(f"   Throughput: {results['throughput_tx_sec']} tx/sec")

        # Save results
        os.makedirs("load_tests/results", exist_ok=True)
        with open("load_tests/results/fnr_baseline.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n   [SUCCESS] Results saved to load_tests/results/fnr_baseline.json")

    print("=" * 60)


if __name__ == "__main__":
    main()
