"""
SentinelAI — Thread-Safe Scoring Wrapper.

Wraps the existing Isolation Forest + Logistic Regression + A* decision
engine behind a clean, stateless interface suitable for concurrent use
by multiple worker processes.

Thread-safety analysis:
    - AnomalyDetector.predict/anomaly_score: sklearn's predict is read-only
      after fit(), internally calls numpy operations — thread-safe ✓
    - FraudClassifier.predict_proba: same as above — thread-safe ✓
    - DecisionState: frozen dataclass — immutable, thread-safe ✓
    - CostFunction.compute: pure function, no state mutation — thread-safe ✓
    - AStarDecisionEngine.search: creates local heap, no shared state — thread-safe ✓

    The only mutable operation is model.fit(), which happens once at
    initialization. After that, all inference operations are read-only.

Usage:
    scorer = ThreadSafeScorer(data_path="data/transactions.csv")
    result = scorer.score(transaction_dict)
"""

import os
import sys
import time
import logging
from typing import Optional

import pandas as pd

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier
from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine
from streaming.models import TransactionMessage, ScoringResult

logger = logging.getLogger(__name__)


class ThreadSafeScorer:
    """Wraps the full scoring pipeline for concurrent worker use.

    Each worker process should create its own ThreadSafeScorer instance.
    After initialization (which loads and trains models), all scoring
    operations are read-only and safe for concurrent use.

    Attributes:
        anomaly_detector: Trained Isolation Forest model.
        classifier: Trained Logistic Regression pipeline.
        cost_function: Cost model for A* decision engine.
        decision_engine: A* search engine.
        worker_id: Identifier for this worker (for tracing).
    """

    def __init__(
        self,
        data_path: str = "data/transactions.csv",
        fraud_loss: float = 1000.0,
        false_positive_cost: float = 50.0,
        review_cost: float = 20.0,
        worker_id: str = "worker-0",
    ) -> None:
        """Initialize and train all models.

        This is the only mutating operation. After __init__ completes,
        the scorer is fully read-only.

        Args:
            data_path: Path to training data CSV.
            fraud_loss: Cost of allowing fraud through.
            false_positive_cost: Cost of blocking legitimate transactions.
            review_cost: Cost of manual review.
            worker_id: Identifier for this worker instance.
        """
        self.worker_id = worker_id

        logger.info(
            "Initializing scorer (worker=%s, data=%s)", worker_id, data_path
        )

        # Load training data
        df = pd.read_csv(data_path)

        # Train models (one-time mutable operation)
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_detector.fit(df)

        self.classifier = FraudClassifier()
        self.classifier.fit(df)

        # Cost function and decision engine (stateless after init)
        self.cost_function = CostFunction(
            fraud_loss=fraud_loss,
            false_positive_cost=false_positive_cost,
            review_cost=review_cost,
        )
        self.decision_engine = AStarDecisionEngine(self.cost_function)

        logger.info("Scorer initialized (worker=%s)", worker_id)

    def score(self, message: TransactionMessage) -> ScoringResult:
        """Score a single transaction through the full pipeline.

        Pipeline:
            1. Convert to DataFrame (for sklearn compatibility)
            2. Compute anomaly score (Isolation Forest)
            3. Compute fraud probability (Logistic Regression)
            4. Run A* decision search
            5. Return ScoringResult

        Args:
            message: Transaction to score.

        Returns:
            ScoringResult: Complete scoring result with decision.
        """
        start_time = time.perf_counter()

        # Convert to DataFrame for sklearn
        tx_df = pd.DataFrame([message.to_feature_dict()])

        # ML inference (read-only, thread-safe)
        anomaly_score = float(self.anomaly_detector.anomaly_score(tx_df)[0])
        fraud_probability = float(self.classifier.predict_proba(tx_df)[0])

        # A* decision search (creates local heap, thread-safe)
        state = DecisionState(
            transaction_id=message.transaction_id,
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
        )
        final_state = self.decision_engine.search(state)

        # Compute cost of chosen action
        expected_cost = self.cost_function.compute(state, final_state.action)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ScoringResult(
            transaction_id=message.transaction_id,
            amount=message.amount,
            anomaly_score=anomaly_score,
            fraud_probability=fraud_probability,
            decision=final_state.action,
            expected_cost=round(expected_cost, 4),
            is_fraud=message.is_fraud,
            worker_id=self.worker_id,
            scored_at=time.time(),
            ingested_at=message.ingested_at,
            scoring_latency_ms=round(elapsed_ms, 3),
        )

    def score_batch(
        self, messages: list[TransactionMessage]
    ) -> list[ScoringResult]:
        """Score a batch of transactions.

        Args:
            messages: List of transactions to score.

        Returns:
            list[ScoringResult]: Scoring results in the same order.
        """
        return [self.score(msg) for msg in messages]
