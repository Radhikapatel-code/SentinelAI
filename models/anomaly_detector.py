import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Detects anomalous transactions using Isolation Forest.
    """

    def __init__(self, contamination=0.2, random_state=42):
        """
        contamination: expected proportion of anomalies
        """
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, data: pd.DataFrame):
        """
        Train the anomaly detection model.
        """
        features = self._select_features(data)
        self.model.fit(features)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly labels.
        Returns:
            -1 → anomaly
             1 → normal
        """
        features = self._select_features(data)
        return self.model.predict(features)

    def anomaly_score(self, data: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores.
        Lower score → more anomalous
        """
        features = self._select_features(data)
        return self.model.decision_function(features)

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select numerical features for anomaly detection.
        """
        return data[
            [
                "amount",
                "transaction_time",
                "location_change",
                "device_change",
                "merchant_risk",
            ]
        ]
