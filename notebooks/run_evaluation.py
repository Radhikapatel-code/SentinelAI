"""
SentinelAI Model Evaluation Script.

Trains and evaluates the fraud detection models on the Kaggle Credit Card
Fraud Detection dataset. Produces:
- Classification report (precision, recall, F1)
- AUC-ROC and Precision-Recall curves (saved as PNG)
- SHAP summary plot (saved as PNG)
- Trained model artifacts (.pkl files)
- README-ready results table

Usage:
    python notebooks/run_evaluation.py

    Or with a custom dataset path:
    python notebooks/run_evaluation.py --data-path data/creditcard.csv

Prerequisites:
    The Kaggle Credit Card Fraud dataset should be placed at
    data/creditcard.csv. Download from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    Alternatively, install kagglehub and set Kaggle credentials:
    pip install kagglehub
"""

import sys
import os
import argparse
import time
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Matplotlib backend for headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the Credit Card Fraud dataset.

    Args:
        data_path: Path to the creditcard.csv file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not data_path.exists():
        print(f"\n❌ Dataset not found at: {data_path}")
        print("\nPlease download the Credit Card Fraud Detection dataset:")
        print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"\nPlace the file at: {data_path}")
        print("\nAlternatively, you can use kagglehub:")
        print("  pip install kagglehub")
        print("  python -c \"import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud')\"")
        sys.exit(1)

    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Fraud rate: {df['Class'].mean() * 100:.3f}%")
    return df


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.002,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest anomaly detector.

    Args:
        X_train: Training feature matrix.
        contamination: Expected proportion of anomalies.
        random_state: Random seed.

    Returns:
        IsolationForest: Trained model.
    """
    print("\n🌲 Training Isolation Forest...")
    start = time.time()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    elapsed = time.time() - start
    print(f"   Training time: {elapsed:.2f}s")
    return model


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """Train a Logistic Regression fraud classifier pipeline.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.

    Returns:
        Pipeline: Fitted sklearn pipeline (scaler + classifier).
    """
    print("\n📊 Training Logistic Regression classifier...")
    start = time.time()
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"   Training time: {elapsed:.2f}s")
    return pipeline


def evaluate_isolation_forest(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the Isolation Forest model.

    Args:
        model: Trained Isolation Forest.
        X_test: Test feature matrix.
        y_test: True labels.

    Returns:
        dict: Evaluation metrics including AUC-ROC.
    """
    print("\n📈 Evaluating Isolation Forest...")

    # Anomaly scores (lower = more anomalous)
    scores = model.decision_function(X_test)
    # Invert so that higher = more anomalous (matches y_test convention)
    anomaly_scores = -scores

    # AUC-ROC
    auc_roc = roc_auc_score(y_test, anomaly_scores)
    print(f"   AUC-ROC: {auc_roc:.4f}")

    # Precision-Recall AUC (more relevant for imbalanced data)
    ap = average_precision_score(y_test, anomaly_scores)
    print(f"   Average Precision (PR-AUC): {ap:.4f}")

    # Binary predictions
    predictions = model.predict(X_test)
    # IsolationForest: -1 = anomaly, 1 = normal → convert to 0/1
    y_pred = (predictions == -1).astype(int)

    print("\n   Classification Report (Isolation Forest):")
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    print("   " + report.replace("\n", "\n   "))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:\n   {cm}")

    return {
        "auc_roc": auc_roc,
        "average_precision": ap,
        "anomaly_scores": anomaly_scores,
        "y_pred": y_pred,
    }


def evaluate_classifier(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate the Logistic Regression classifier.

    Args:
        pipeline: Trained classifier pipeline.
        X_test: Test feature matrix.
        y_test: True labels.

    Returns:
        dict: Evaluation metrics.
    """
    print("\n📈 Evaluating Logistic Regression classifier...")

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   Average Precision (PR-AUC): {ap:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    print("\n   Classification Report (Logistic Regression):")
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    print("   " + report.replace("\n", "\n   "))

    # Precision at 90% recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    # Find precision where recall >= 0.90
    mask = recalls >= 0.90
    if mask.any():
        precision_at_90_recall = precisions[mask][-1]
        print(f"   Precision @ 90% Recall: {precision_at_90_recall:.4f}")
    else:
        precision_at_90_recall = 0.0
        print("   Precision @ 90% Recall: N/A (model cannot reach 90% recall)")

    return {
        "auc_roc": auc_roc,
        "average_precision": ap,
        "f1": f1,
        "precision_at_90_recall": precision_at_90_recall,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def plot_roc_curves(
    y_test: np.ndarray,
    if_scores: np.ndarray,
    lr_proba: np.ndarray,
    save_path: Path,
) -> None:
    """Plot ROC curves for both models.

    Args:
        y_test: True labels.
        if_scores: Isolation Forest anomaly scores.
        lr_proba: Logistic Regression probabilities.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Isolation Forest
    fpr_if, tpr_if, _ = roc_curve(y_test, if_scores)
    auc_if = roc_auc_score(y_test, if_scores)
    ax.plot(fpr_if, tpr_if, label=f"Isolation Forest (AUC={auc_if:.3f})", linewidth=2)

    # Logistic Regression
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    auc_lr = roc_auc_score(y_test, lr_proba)
    ax.plot(
        fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})", linewidth=2
    )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — SentinelAI Fraud Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 ROC curve saved to: {save_path}")


def plot_precision_recall_curve(
    y_test: np.ndarray,
    if_scores: np.ndarray,
    lr_proba: np.ndarray,
    save_path: Path,
) -> None:
    """Plot Precision-Recall curves for both models.

    Args:
        y_test: True labels.
        if_scores: Isolation Forest anomaly scores.
        lr_proba: Logistic Regression probabilities.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Isolation Forest
    prec_if, rec_if, _ = precision_recall_curve(y_test, if_scores)
    ap_if = average_precision_score(y_test, if_scores)
    ax.plot(rec_if, prec_if, label=f"Isolation Forest (AP={ap_if:.3f})", linewidth=2)

    # Logistic Regression
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_proba)
    ap_lr = average_precision_score(y_test, lr_proba)
    ax.plot(
        rec_lr, prec_lr, label=f"Logistic Regression (AP={ap_lr:.3f})", linewidth=2
    )

    # Baseline (fraud rate)
    fraud_rate = y_test.mean()
    ax.axhline(y=fraud_rate, color="k", linestyle="--", alpha=0.3, label=f"Baseline ({fraud_rate:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — SentinelAI", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 PR curve saved to: {save_path}")


def generate_shap_plot(
    pipeline: Pipeline,
    X_test: np.ndarray,
    feature_names: list[str],
    save_path: Path,
) -> None:
    """Generate and save a SHAP summary plot.

    Args:
        pipeline: Trained classifier pipeline.
        X_test: Test feature matrix (subset for speed).
        feature_names: Names of the features.
        save_path: Path to save the plot.
    """
    try:
        import shap
    except ImportError:
        print("⚠️  SHAP not installed, skipping SHAP plot")
        return

    print("\n🔍 Generating SHAP summary plot...")

    # Use a small background sample for speed
    background = X_test[:100]
    sample = X_test[:500]

    def model_predict(X: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(X)[:, 1]

    explainer = shap.Explainer(model_predict, background, feature_names=feature_names)
    shap_values = explainer(sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance — SentinelAI", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 SHAP plot saved to: {save_path}")


def save_model_artifacts(
    isolation_forest: IsolationForest,
    classifier_pipeline: Pipeline,
    artifacts_dir: Path,
) -> None:
    """Save trained models as pickle artifacts.

    Args:
        isolation_forest: Trained Isolation Forest model.
        classifier_pipeline: Trained classifier pipeline.
        artifacts_dir: Directory to save the artifacts.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if_path = artifacts_dir / "isolation_forest.pkl"
    clf_path = artifacts_dir / "classifier.pkl"

    with open(if_path, "wb") as f:
        pickle.dump(isolation_forest, f)
    print(f"\n💾 Isolation Forest saved to: {if_path}")

    with open(clf_path, "wb") as f:
        pickle.dump(classifier_pipeline, f)
    print(f"💾 Classifier saved to: {clf_path}")


def print_readme_table(
    if_metrics: dict[str, Any],
    lr_metrics: dict[str, Any],
    dataset_size: int,
    fraud_rate: float,
    decision_time_ms: float,
) -> None:
    """Print a Markdown-formatted results table for the README.

    Args:
        if_metrics: Isolation Forest evaluation metrics.
        lr_metrics: Logistic Regression evaluation metrics.
        dataset_size: Total number of transactions.
        fraud_rate: Percentage of fraudulent transactions.
        decision_time_ms: Average decision time in milliseconds.
    """
    print("\n" + "=" * 60)
    print("📋 README Results Table (copy this to your README.md)")
    print("=" * 60)
    print()
    print("| Metric | Isolation Forest | Logistic Regression |")
    print("|--------|-----------------|-------------------|")
    print(f"| AUC-ROC | {if_metrics['auc_roc']:.4f} | {lr_metrics['auc_roc']:.4f} |")
    print(f"| Avg Precision (PR-AUC) | {if_metrics['average_precision']:.4f} | {lr_metrics['average_precision']:.4f} |")
    print(f"| F1 Score | — | {lr_metrics['f1']:.4f} |")
    print(f"| Precision @90% Recall | — | {lr_metrics['precision_at_90_recall']:.4f} |")
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Dataset Size | {dataset_size:,} transactions |")
    print(f"| Fraud Rate | {fraud_rate:.3f}% |")
    print(f"| Avg Decision Time | {decision_time_ms:.1f}ms |")
    print()


def measure_decision_time(
    pipeline: Pipeline,
    X_sample: np.ndarray,
    n_iterations: int = 1000,
) -> float:
    """Measure average prediction time in milliseconds.

    Args:
        pipeline: Trained classifier pipeline.
        X_sample: A single sample to predict on.
        n_iterations: Number of predictions to average over.

    Returns:
        float: Average prediction time in milliseconds.
    """
    single = X_sample[:1]
    # Warmup
    for _ in range(10):
        pipeline.predict_proba(single)

    start = time.time()
    for _ in range(n_iterations):
        pipeline.predict_proba(single)
    elapsed = (time.time() - start) / n_iterations * 1000
    return elapsed


def main() -> None:
    """Run the full evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="SentinelAI Model Evaluation"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to creditcard.csv",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 SentinelAI — Model Evaluation")
    print("=" * 60)

    # Load dataset
    data_path = Path(args.data_path)
    df = load_dataset(data_path)

    # Prepare features
    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]
    X = df[feature_cols].values
    y = df["Class"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n📊 Split: {len(X_train):,} train / {len(X_test):,} test")

    # Train models
    iso_forest = train_isolation_forest(X_train, contamination=0.002)
    classifier = train_classifier(X_train, y_train)

    # Evaluate
    if_metrics = evaluate_isolation_forest(iso_forest, X_test, y_test)
    lr_metrics = evaluate_classifier(classifier, X_test, y_test)

    # Create output directories
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot curves
    plot_roc_curves(
        y_test,
        if_metrics["anomaly_scores"],
        lr_metrics["y_proba"],
        ASSETS_DIR / "roc_curves.png",
    )
    plot_precision_recall_curve(
        y_test,
        if_metrics["anomaly_scores"],
        lr_metrics["y_proba"],
        ASSETS_DIR / "pr_curves.png",
    )

    # SHAP plot
    generate_shap_plot(
        classifier,
        X_test,
        feature_cols,
        ASSETS_DIR / "shap_summary.png",
    )

    # Save model artifacts
    save_model_artifacts(iso_forest, classifier, ARTIFACTS_DIR)

    # Measure decision time
    decision_time = measure_decision_time(classifier, X_test)
    print(f"\n⏱️  Average decision time: {decision_time:.2f}ms")

    # Print README table
    print_readme_table(
        if_metrics=if_metrics,
        lr_metrics=lr_metrics,
        dataset_size=len(df),
        fraud_rate=df["Class"].mean() * 100,
        decision_time_ms=decision_time,
    )

    print("\n✅ Evaluation complete!")
    print(f"   Assets saved to: {ASSETS_DIR}")
    print(f"   Models saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
