"""
Shared pytest fixtures for SentinelAI test suite.

Provides pre-configured instances of all core components:
anomaly detector, classifier, decision engine, and explainer.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier
from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine
from explainability.shap_explainer import ShapExplainer


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame matching the transactions.csv schema.

    Returns:
        pd.DataFrame: 10-row DataFrame with realistic transaction data,
        including both legitimate and fraudulent transactions.
    """
    return pd.DataFrame(
        {
            "transaction_id": list(range(1, 11)),
            "amount": [
                1200, 45000, 300, 89000, 1500,
                67000, 220, 54000, 1800, 76000,
            ],
            "transaction_time": [14, 2, 10, 1, 16, 23, 11, 3, 18, 0],
            "location_change": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "device_change": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            "merchant_risk": [
                0.1, 0.8, 0.05, 0.9, 0.2,
                0.7, 0.03, 0.85, 0.15, 0.95,
            ],
            "is_fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def trained_anomaly_detector(sample_dataframe: pd.DataFrame) -> AnomalyDetector:
    """Fit an AnomalyDetector on the sample data.

    Args:
        sample_dataframe: The sample transaction DataFrame.

    Returns:
        AnomalyDetector: A trained anomaly detector instance.
    """
    detector = AnomalyDetector(contamination=0.2, random_state=42)
    detector.fit(sample_dataframe)
    return detector


@pytest.fixture
def trained_classifier(sample_dataframe: pd.DataFrame) -> FraudClassifier:
    """Fit a FraudClassifier on the sample data.

    Args:
        sample_dataframe: The sample transaction DataFrame.

    Returns:
        FraudClassifier: A trained classifier instance.
    """
    classifier = FraudClassifier()
    classifier.fit(sample_dataframe)
    return classifier


@pytest.fixture
def cost_function() -> CostFunction:
    """Create a CostFunction with default weights.

    Returns:
        CostFunction: Instance with fraud_loss=1000,
        false_positive_cost=50, review_cost=20.
    """
    return CostFunction(
        fraud_loss=1000,
        false_positive_cost=50,
        review_cost=20,
    )


@pytest.fixture
def decision_engine(cost_function: CostFunction) -> AStarDecisionEngine:
    """Create an AStarDecisionEngine with the default cost function.

    Args:
        cost_function: The CostFunction fixture.

    Returns:
        AStarDecisionEngine: Ready-to-use decision engine.
    """
    return AStarDecisionEngine(cost_function)


@pytest.fixture
def shap_explainer(trained_classifier: FraudClassifier) -> ShapExplainer:
    """Create a ShapExplainer backed by the trained classifier.

    Args:
        trained_classifier: The trained FraudClassifier fixture.

    Returns:
        ShapExplainer: Ready-to-use SHAP explainer.
    """
    return ShapExplainer(trained_classifier.pipeline)


@pytest.fixture
def legitimate_transaction() -> pd.DataFrame:
    """A clearly legitimate transaction (low amount, no flags).

    Returns:
        pd.DataFrame: Single-row DataFrame representing a normal transaction.
    """
    return pd.DataFrame(
        [
            {
                "transaction_id": 99,
                "amount": 500,
                "transaction_time": 14,
                "location_change": 0,
                "device_change": 0,
                "merchant_risk": 0.05,
            }
        ]
    )


@pytest.fixture
def fraudulent_transaction() -> pd.DataFrame:
    """A clearly fraudulent transaction (high amount, all red flags).

    Returns:
        pd.DataFrame: Single-row DataFrame representing a suspicious transaction.
    """
    return pd.DataFrame(
        [
            {
                "transaction_id": 100,
                "amount": 92000,
                "transaction_time": 2,
                "location_change": 1,
                "device_change": 1,
                "merchant_risk": 0.95,
            }
        ]
    )
