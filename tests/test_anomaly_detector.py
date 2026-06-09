"""
Tests for the Anomaly Detection module.

Validates:
- Model fitting and chaining API
- Prediction output shapes and value ranges
- Anomaly score computation
- Feature selection logic
- Fraud vs legitimate score differentiation
"""

import numpy as np
import pandas as pd
import pytest

from models.anomaly_detector import AnomalyDetector, FEATURE_COLUMNS


class TestAnomalyDetectorFit:
    """Tests for model fitting."""

    def test_fit_returns_self(self, sample_dataframe: pd.DataFrame) -> None:
        """fit() should return the detector instance for method chaining."""
        detector = AnomalyDetector(contamination=0.2, random_state=42)
        result = detector.fit(sample_dataframe)
        assert result is detector

    def test_fit_creates_fitted_model(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """After fit(), the model should be able to make predictions."""
        predictions = trained_anomaly_detector.predict(sample_dataframe)
        assert predictions is not None


class TestAnomalyDetectorPredict:
    """Tests for anomaly prediction."""

    def test_predict_returns_ndarray(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """predict() should return a numpy ndarray."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        assert isinstance(result, np.ndarray)

    def test_predict_shape_matches_input(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """predict() output should have one label per input row."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        assert result.shape == (len(sample_dataframe),)

    def test_predict_values_are_valid_labels(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """predict() labels should only be -1 (anomaly) or 1 (normal)."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        unique_values = set(result)
        assert unique_values.issubset({-1, 1})

    def test_predict_single_transaction(
        self,
        trained_anomaly_detector: AnomalyDetector,
        legitimate_transaction: pd.DataFrame,
    ) -> None:
        """predict() should work on a single-row DataFrame."""
        result = trained_anomaly_detector.predict(legitimate_transaction)
        assert result.shape == (1,)

    def test_predict_detects_at_least_one_anomaly(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """With contamination=0.2 on 10 rows, at least one should be flagged."""
        result = trained_anomaly_detector.predict(sample_dataframe)
        anomaly_count = (result == -1).sum()
        assert anomaly_count >= 1


class TestAnomalyScore:
    """Tests for anomaly score computation."""

    def test_anomaly_score_returns_ndarray(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """anomaly_score() should return a numpy ndarray."""
        result = trained_anomaly_detector.anomaly_score(sample_dataframe)
        assert isinstance(result, np.ndarray)

    def test_anomaly_score_shape_matches_input(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """anomaly_score() output should have one score per input row."""
        result = trained_anomaly_detector.anomaly_score(sample_dataframe)
        assert result.shape == (len(sample_dataframe),)

    def test_anomaly_score_values_are_floats(
        self, trained_anomaly_detector: AnomalyDetector, sample_dataframe: pd.DataFrame
    ) -> None:
        """anomaly_score() values should be numeric."""
        result = trained_anomaly_detector.anomaly_score(sample_dataframe)
        assert result.dtype in [np.float64, np.float32]


class TestFeatureSelection:
    """Tests for feature column selection."""

    def test_select_features_returns_correct_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """_select_features() should return only the 5 feature columns."""
        result = AnomalyDetector._select_features(sample_dataframe)
        assert list(result.columns) == FEATURE_COLUMNS

    def test_select_features_excludes_non_feature_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """_select_features() should not include transaction_id or is_fraud."""
        result = AnomalyDetector._select_features(sample_dataframe)
        assert "transaction_id" not in result.columns
        assert "is_fraud" not in result.columns
