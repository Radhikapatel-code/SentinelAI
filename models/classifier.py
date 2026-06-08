"""
SentinelAI Fraud Classifier Module.

Provides probability-based fraud classification using a Logistic Regression
pipeline with StandardScaler preprocessing. Uses class_weight='balanced'
to handle the inherent class imbalance in fraud detection datasets.

Example:
    >>> classifier = FraudClassifier()
    >>> classifier.fit(training_data)
    >>> probabilities = classifier.predict_proba(new_transactions)
    >>> labels = classifier.predict(new_transactions)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Feature columns used by the classifier
FEATURE_COLUMNS: list[str] = [
    "amount",
    "transaction_time",
    "location_change",
    "device_change",
    "merchant_risk",
]


class FraudClassifier:
    """Probability-based fraud classifier using Logistic Regression.

    Wraps a sklearn Pipeline (StandardScaler → LogisticRegression) to
    produce calibrated fraud probabilities for each transaction.

    Attributes:
        pipeline: The sklearn Pipeline containing scaler and classifier.
    """

    def __init__(self) -> None:
        """Initialize the classifier pipeline with balanced class weights.

        Uses 'liblinear' solver which is efficient for small datasets
        and supports L1/L2 regularization.
        """
        self.pipeline: Pipeline = Pipeline(
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

    def fit(self, data: pd.DataFrame) -> "FraudClassifier":
        """Train the fraud classifier on labeled transaction data.

        Args:
            data: DataFrame containing transaction features and an
                'is_fraud' column (0 = legitimate, 1 = fraudulent).

        Returns:
            FraudClassifier: The fitted classifier instance (for chaining).
        """
        X, y = self._prepare_data(data)
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Predict the probability of fraud for each transaction.

        Args:
            data: DataFrame containing transaction features.

        Returns:
            np.ndarray: Array of fraud probabilities (float, 0.0 to 1.0).
                Each value represents P(fraud | features).
        """
        X = self._select_features(data)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict binary fraud labels for each transaction.

        Args:
            data: DataFrame containing transaction features.

        Returns:
            np.ndarray: Array of predicted labels (0 or 1).
        """
        X = self._select_features(data)
        return self.pipeline.predict(X)

    def _prepare_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and labels from the training data.

        Args:
            data: DataFrame with features and 'is_fraud' label column.

        Returns:
            tuple: (X features DataFrame, y labels Series).
        """
        X = self._select_features(data)
        y = data["is_fraud"]
        return X, y

    @staticmethod
    def _select_features(data: pd.DataFrame) -> pd.DataFrame:
        """Select features used by the classifier.

        Args:
            data: Raw transaction DataFrame.

        Returns:
            pd.DataFrame: DataFrame with only the required feature columns.
        """
        return data[FEATURE_COLUMNS]
