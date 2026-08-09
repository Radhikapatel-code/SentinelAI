"""
SentinelAI — Scoring Result Models.

Defines the data structures passed between streaming components:
    Producer → Consumer → Scorer → Sink

All models are frozen dataclasses for immutability and thread safety.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class TransactionMessage:
    """A transaction as received from the Kafka/Redpanda topic.

    Attributes:
        transaction_id: Unique identifier.
        amount: Transaction amount in currency units.
        transaction_time: Hour of transaction (0–23).
        location_change: Whether location changed (0 or 1).
        device_change: Whether device changed (0 or 1).
        merchant_risk: Merchant risk score (0.0–1.0).
        is_fraud: Ground-truth label for FNR measurement (optional).
        ingested_at: Timestamp when the producer sent this message.
    """

    transaction_id: int
    amount: float
    transaction_time: int
    location_change: int
    device_change: int
    merchant_risk: float
    is_fraud: Optional[bool] = None
    ingested_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize to JSON string for Kafka publishing."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "TransactionMessage":
        """Deserialize from JSON string.

        Args:
            data: JSON string representation.

        Returns:
            TransactionMessage: Deserialized instance.
        """
        d = json.loads(data)
        return cls(**d)

    def to_feature_dict(self) -> dict:
        """Extract model-compatible feature dictionary.

        Returns:
            dict: Features matching FEATURE_COLUMNS in the ML models.
        """
        return {
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "transaction_time": self.transaction_time,
            "location_change": self.location_change,
            "device_change": self.device_change,
            "merchant_risk": self.merchant_risk,
        }


@dataclass(frozen=True)
class ScoringResult:
    """Result of scoring a single transaction.

    Attributes:
        transaction_id: Unique identifier.
        amount: Original transaction amount.
        anomaly_score: Isolation Forest anomaly score.
        fraud_probability: Logistic Regression fraud probability.
        decision: A* optimal action ("approve", "block", "review").
        expected_cost: Cost of the chosen action.
        is_fraud: Ground-truth label (for FNR measurement).
        worker_id: ID of the worker that scored this transaction.
        scored_at: Timestamp when scoring completed.
        ingested_at: Timestamp when the producer sent this message.
        scoring_latency_ms: Time taken to score (milliseconds).
    """

    transaction_id: int
    amount: float
    anomaly_score: float
    fraud_probability: float
    decision: str
    expected_cost: float
    is_fraud: Optional[bool] = None
    worker_id: str = ""
    scored_at: float = field(default_factory=time.time)
    ingested_at: float = 0.0
    scoring_latency_ms: float = 0.0

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "ScoringResult":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(**d)

    def to_db_tuple(self) -> tuple:
        """Convert to tuple for PostgreSQL batch insert.

        Returns:
            tuple: Values matching scored_transactions table columns.
        """
        return (
            self.transaction_id,
            self.amount,
            self.anomaly_score,
            self.fraud_probability,
            self.decision,
            self.expected_cost,
            self.is_fraud,
            self.worker_id,
            self.scored_at,
            self.ingested_at,
        )
