"""
Tests for the AnomalyDetector (Isolation Forest).

Validates that the anomaly detection model:
- Trains without errors
- Returns correct output shapes and types
- Distinguishes between normal and fraudulent transactions
"""

import numpy as np
import pandas as pd
import pytest

from models.anomaly_detector import AnomalyDetector


class TestAnomalyDetectorFit:
    """Tests for model training."""

    def test_fit_does_not_raise(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Training on valid data should complete without errors."""
        detector = AnomalyDetector(contamination=0.2, random_state=42)
        detector.fit(sample_dataframe)  # Should not raise

    def test_fit_with_different_contamination(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Training with a custom contamination rate should work."""
        detector = AnomalyDetector(contamination=0.1, random_state=42)
        detector.fit(sample_dataframe)  # Should not raise


class TestAnomalyDetectorPredict:
    """Tests for anomaly label prediction."""

    def test_predict_returns_array(
        self,
        trained_anomaly_detector: AnomalyDetector,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Predict should return a numpy array."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        assert isinstance(result, np.ndarray)

    def test_predict_returns_correct_length(
        self,
        trained_anomaly_detector: AnomalyDetector,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Predict should return one label per input row."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        assert len(result) == len(sample_dataframe)

    def test_predict_returns_valid_labels(
        self,
        trained_anomaly_detector: AnomalyDetector,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Labels should only be -1 (anomaly) or 1 (normal)."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        assert set(result).issubset({-1, 1})


class TestAnomalyDetectorScore:
    """Tests for anomaly scoring."""

    def test_anomaly_score_returns_array(
        self,
        trained_anomaly_detector: AnomalyDetector,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Anomaly scores should be a numpy array of floats."""
        scores = trained_anomaly_detector.anomaly_score(sample_dataframe)
        assert isinstance(scores, np.ndarray)
        assert scores.dtype in [np.float64, np.float32]

    def test_anomaly_score_returns_correct_length(
        self,
        trained_anomaly_detector: AnomalyDetector,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Should return one score per input row."""
        scores = trained_anomaly_detector.anomaly_score(sample_dataframe)
        assert len(scores) == len(sample_dataframe)

    def test_flags_synthetic_fraud(
        self,
        trained_anomaly_detector: AnomalyDetector,
        legitimate_transaction: pd.DataFrame,
        fraudulent_transaction: pd.DataFrame,
    ) -> None:
        """A high-risk transaction should get a lower (more anomalous) score
        than a clearly legitimate transaction.

        Isolation Forest convention: lower score = more anomalous.
        """
        legit_score = trained_anomaly_detector.anomaly_score(
            legitimate_transaction
        )[0]
        fraud_score = trained_anomaly_detector.anomaly_score(
            fraudulent_transaction
        )[0]
        assert fraud_score < legit_score, (
            f"Fraud score ({fraud_score:.4f}) should be lower than "
            f"legit score ({legit_score:.4f})"
        )
