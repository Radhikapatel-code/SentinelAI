"""
SentinelAI Models Package.

Provides machine learning components for fraud detection:
- AnomalyDetector: Isolation Forest-based anomaly detection
- FraudClassifier: Logistic Regression-based fraud probability scoring
"""

from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier

__all__ = ["AnomalyDetector", "FraudClassifier"]
