import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class FraudClassifier:
    """
    Probability-based fraud classifier using Logistic Regression.
    """

    def __init__(self):
        """
        Initializes a pipeline with feature scaling and logistic regression.
        """
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, data: pd.DataFrame):
        """
        Train the fraud classifier.
        """
        X, y = self._prepare_data(data)
        self.pipeline.fit(X, y)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of fraud.
        Returns probability of class '1' (fraud).
        """
        X = self._select_features(data)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud class.
        """
        X = self._select_features(data)
        return self.pipeline.predict(X)

    def _prepare_data(self, data: pd.DataFrame):
        """
        Separate features and labels.
        """
        X = self._select_features(data)
        y = data["is_fraud"]
        return X, y

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select features used by the classifier.
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
