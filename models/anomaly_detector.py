"""
SentinelAI Anomaly Detection Module.

Uses Isolation Forest to detect anomalous financial transactions.
The model learns normal transaction patterns and flags statistically
rare transactions as potential fraud.

Isolation Forest works by randomly partitioning data; anomalies
require fewer partitions to isolate, resulting in shorter path lengths
and lower anomaly scores.

Example:
    >>> detector = AnomalyDetector(contamination=0.2)
    >>> detector.fit(training_data)
    >>> scores = detector.anomaly_score(new_transactions)
    >>> predictions = detector.predict(new_transactions)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Feature columns used by the anomaly detector
FEATURE_COLUMNS: list[str] = [
    "amount",
    "transaction_time",
    "location_change",
    "device_change",
    "merchant_risk",
]


class AnomalyDetector:
    """Detects anomalous transactions using Isolation Forest.

    The detector operates on 5 numerical features and produces both
    binary anomaly labels and continuous anomaly scores.

    Attributes:
        model: The fitted Isolation Forest model.
    """

    def __init__(
        self,
        contamination: float = 0.2,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in the dataset.
                Range: (0.0, 0.5]. Higher values flag more transactions.
            n_estimators: Number of isolation trees in the ensemble.
            random_state: Random seed for reproducibility.
        """
        self.model: IsolationForest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, data: pd.DataFrame) -> "AnomalyDetector":
        """Train the anomaly detection model on historical transactions.

        Args:
            data: DataFrame containing transaction data with the required
                feature columns.

        Returns:
            AnomalyDetector: The fitted detector instance (for chaining).
        """
        features = self._select_features(data)
        self.model.fit(features)
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict anomaly labels for transactions.

        Args:
            data: DataFrame containing transaction features.

        Returns:
            np.ndarray: Array of labels where:
                -1 = anomaly (potentially fraudulent)
                 1 = normal (legitimate)
        """
        features = self._select_features(data)
        return self.model.predict(features)

    def anomaly_score(self, data: pd.DataFrame) -> np.ndarray:
        """Compute anomaly scores for transactions.

        Uses the Isolation Forest decision function. Lower scores indicate
        more anomalous transactions (shorter average path length in the
        isolation trees).

        Args:
            data: DataFrame containing transaction features.

        Returns:
            np.ndarray: Array of anomaly scores (float).
                Lower score = more anomalous.
        """
        features = self._select_features(data)
        return self.model.decision_function(features)

    @staticmethod
    def _select_features(data: pd.DataFrame) -> pd.DataFrame:
        """Select numerical features for anomaly detection.

        Args:
            data: Raw transaction DataFrame.

        Returns:
            pd.DataFrame: DataFrame with only the required feature columns.
        """
        return data[FEATURE_COLUMNS]
