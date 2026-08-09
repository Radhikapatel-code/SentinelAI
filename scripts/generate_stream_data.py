"""
SentinelAI — Synthetic Streaming Data Generator.

Generates a large synthetic transaction dataset matching the 5-feature
schema used by the production models. Includes ground-truth `is_fraud`
labels for false-negative-rate measurement under load.

Usage:
    python scripts/generate_stream_data.py
    python scripts/generate_stream_data.py --count 500000 --output data/stream_transactions.csv

Why synthetic and not the creditcard.csv PCA features:
    The production models (AnomalyDetector, FraudClassifier) expect the
    5 named features: amount, transaction_time, location_change,
    device_change, merchant_risk. The Kaggle creditcard.csv uses V1-V28
    PCA features with different distributions. Rather than building a
    fragile PCA→feature mapping, we generate synthetic data with realistic
    distributions that match the model's training schema.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def generate_transactions(
    count: int = 100000,
    fraud_rate: float = 0.017,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic transactions with realistic distributions.

    Args:
        count: Number of transactions to generate.
        fraud_rate: Fraction of fraudulent transactions (0.017 ≈ 1.7%).
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Transaction data with columns:
            transaction_id, amount, transaction_time, location_change,
            device_change, merchant_risk, is_fraud
    """
    rng = np.random.default_rng(seed)

    n_fraud = int(count * fraud_rate)
    n_legit = count - n_fraud

    # ── Legitimate transactions ──
    legit = pd.DataFrame({
        "transaction_id": range(1, n_legit + 1),
        "amount": rng.lognormal(mean=6.0, sigma=1.2, size=n_legit).clip(1, 50000),
        "transaction_time": rng.integers(6, 23, size=n_legit),  # daytime bias
        "location_change": rng.choice([0, 1], size=n_legit, p=[0.9, 0.1]),
        "device_change": rng.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        "merchant_risk": rng.beta(2, 10, size=n_legit).clip(0, 1),  # skewed low
        "is_fraud": 0,
    })

    # ── Fraudulent transactions ──
    fraud = pd.DataFrame({
        "transaction_id": range(n_legit + 1, count + 1),
        "amount": rng.lognormal(mean=8.5, sigma=1.5, size=n_fraud).clip(100, 200000),
        "transaction_time": rng.integers(0, 24, size=n_fraud),  # uniform (incl. late night)
        "location_change": rng.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        "device_change": rng.choice([0, 1], size=n_fraud, p=[0.4, 0.6]),
        "merchant_risk": rng.beta(8, 3, size=n_fraud).clip(0, 1),  # skewed high
        "is_fraud": 1,
    })

    # Combine and shuffle
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Reassign sequential transaction IDs after shuffle
    df["transaction_id"] = range(1, len(df) + 1)

    # Round floats for cleaner output
    df["amount"] = df["amount"].round(2)
    df["merchant_risk"] = df["merchant_risk"].round(4)

    return df


def main() -> None:
    """CLI entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for streaming replay"
    )
    parser.add_argument(
        "--count", type=int, default=100000,
        help="Number of transactions to generate (default: 100000)"
    )
    parser.add_argument(
        "--fraud-rate", type=float, default=0.017,
        help="Fraction of fraudulent transactions (default: 0.017)"
    )
    parser.add_argument(
        "--output", type=str, default="data/stream_transactions.csv",
        help="Output CSV path (default: data/stream_transactions.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    print(f"[*] Generating {args.count:,} synthetic transactions "
          f"(fraud rate: {args.fraud_rate:.1%})...")

    df = generate_transactions(
        count=args.count,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df.to_csv(args.output, index=False)

    # Summary
    fraud_count = df["is_fraud"].sum()
    print(f"[OK] Generated {len(df):,} transactions -> {args.output}")
    print(f"   Legitimate: {len(df) - fraud_count:,} ({(1 - fraud_count/len(df)):.1%})")
    print(f"   Fraudulent: {fraud_count:,} ({fraud_count/len(df):.1%})")
    print(f"   Amount range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    print(f"   Avg merchant risk: {df['merchant_risk'].mean():.4f}")


if __name__ == "__main__":
    main()
