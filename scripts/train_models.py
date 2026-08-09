"""
SentinelAI — Offline Model Training Script.

Trains AnomalyDetector (Isolation Forest) and FraudClassifier (Logistic Regression)
and serializes them to `models/artifacts/` using joblib.

This avoids re-training models on worker process startup, turning startup
from CPU-bound training into fast joblib deserialization.

Usage:
    python scripts/train_models.py
"""

import os
import sys
import logging
import joblib
import pandas as pd

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_and_save_models(
    data_path: str = "data/transactions.csv",
    output_dir: str = "models/artifacts",
) -> tuple[str, str]:
    """Train AnomalyDetector and FraudClassifier and save to disk.

    Args:
        data_path: Path to CSV dataset.
        output_dir: Directory where .pkl artifacts will be written.

    Returns:
        tuple[str, str]: Paths to saved (anomaly_detector, classifier) files.
    """
    if not os.path.exists(data_path):
        fallback = "data/creditcard.csv"
        if os.path.exists(fallback):
            data_path = fallback
        else:
            raise FileNotFoundError(f"Training dataset not found at {data_path} or {fallback}")

    logger.info("Loading training data from %s...", data_path)
    df = pd.read_csv(data_path)

    os.makedirs(output_dir, exist_ok=True)
    ad_path = os.path.join(output_dir, "anomaly_detector.pkl")
    clf_path = os.path.join(output_dir, "classifier.pkl")

    logger.info("Fitting AnomalyDetector (Isolation Forest)...")
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(df)
    joblib.dump(anomaly_detector, ad_path)
    logger.info("Saved AnomalyDetector to %s", ad_path)

    logger.info("Fitting FraudClassifier (Logistic Regression)...")
    classifier = FraudClassifier()
    classifier.fit(df)
    joblib.dump(classifier, clf_path)
    logger.info("Saved FraudClassifier to %s", clf_path)

    return ad_path, clf_path


if __name__ == "__main__":
    train_and_save_models()
