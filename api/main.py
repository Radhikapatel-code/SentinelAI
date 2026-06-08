"""
SentinelAI FastAPI Backend.

Provides a REST API for real-time transaction fraud analysis.
The /analyze endpoint runs the full pipeline:
    1. ML scoring (anomaly detection + fraud probability)
    2. A* decision search (cost-optimized action selection)
    3. Explainability (SHAP + optional LLM)

Usage:
    uvicorn api.main:app --reload

API Documentation:
    http://localhost:8000/docs  (Swagger UI — auto-generated)
"""

import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import get_config, AppConfig
from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier
from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine
from explainability.shap_explainer import ShapExplainer
from explainability.llm_explainer import LLMExplainer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Load Configuration
# ──────────────────────────────────────────────
config: AppConfig = get_config()

# ──────────────────────────────────────────────
# Initialize App
# ──────────────────────────────────────────────
app = FastAPI(
    title="SentinelAI API",
    description=(
        "Explainable AI Decision Engine for Intelligent Fraud Detection. "
        "Combines ML anomaly detection, cost-aware A* search, and "
        "SHAP-based explainability."
    ),
    version="0.2.0",
)

# ──────────────────────────────────────────────
# Load Data & Train Models (once at startup)
# ──────────────────────────────────────────────
df: pd.DataFrame = pd.read_csv(config.api.data_path)

anomaly_detector = AnomalyDetector(
    contamination=config.model.contamination,
    n_estimators=config.model.n_estimators,
    random_state=config.model.random_state,
)
anomaly_detector.fit(df)

classifier = FraudClassifier()
classifier.fit(df)

cost_function = CostFunction(
    fraud_loss=config.cost.fraud_loss,
    false_positive_cost=config.cost.false_positive_cost,
    review_cost=config.cost.review_cost,
)
decision_engine = AStarDecisionEngine(cost_function)

explainer = ShapExplainer(classifier.pipeline)

llm_explainer = LLMExplainer(
    api_key=config.llm.api_key,
    model=config.llm.model,
    max_tokens=config.llm.max_tokens,
    enabled=config.llm.enabled,
)


# ──────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────
class TransactionInput(BaseModel):
    """Input schema for transaction analysis.

    All fields are required and represent the features used by
    the ML models for fraud scoring.
    """

    transaction_id: int = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in currency units")
    transaction_time: int = Field(
        ..., ge=0, le=23, description="Hour of transaction (0-23)"
    )
    location_change: int = Field(
        ..., ge=0, le=1, description="Whether location changed (0 or 1)"
    )
    device_change: int = Field(
        ..., ge=0, le=1, description="Whether device changed (0 or 1)"
    )
    merchant_risk: float = Field(
        ..., ge=0.0, le=1.0, description="Merchant risk score (0.0 to 1.0)"
    )


class TransactionResponse(BaseModel):
    """Response schema for transaction analysis."""

    transaction_id: int
    fraud_probability: float
    anomaly_score: float
    decision: str
    explanation: str
    feature_contributions: dict[str, float]


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────
@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict: Status and version information.
    """
    return {"status": "healthy", "version": "0.2.0"}


# ──────────────────────────────────────────────
# Main Analysis Endpoint
# ──────────────────────────────────────────────
@app.post("/analyze", response_model=TransactionResponse)
def analyze_transaction(transaction: TransactionInput) -> dict[str, Any]:
    """Analyze a transaction for fraud using the full SentinelAI pipeline.

    Pipeline:
        1. Convert input to DataFrame
        2. Compute fraud probability (Logistic Regression)
        3. Compute anomaly score (Isolation Forest)
        4. Run A* decision search (cost-optimized)
        5. Generate SHAP explanation
        6. Optionally enhance with LLM explanation

    Args:
        transaction: The transaction data to analyze.

    Returns:
        dict: Analysis results including decision, scores, and explanation.
    """
    # Convert input to DataFrame
    tx_df: pd.DataFrame = pd.DataFrame([transaction.model_dump()])

    # ML Predictions
    fraud_probability: float = float(classifier.predict_proba(tx_df)[0])
    anomaly_score: float = float(anomaly_detector.anomaly_score(tx_df)[0])

    # Decision State
    state = DecisionState(
        transaction_id=transaction.transaction_id,
        fraud_probability=fraud_probability,
        anomaly_score=anomaly_score,
    )

    # Run Decision Engine
    final_state: DecisionState = decision_engine.search(state)

    # Explainability
    shap_values: dict[str, float] = explainer.explain(tx_df)
    shap_explanation: str = explainer.generate_text_explanation(
        shap_values, tx_df
    )

    # Try LLM explanation (falls back to SHAP text if unavailable)
    explanation_text: str = llm_explainer.explain(
        shap_values=shap_values,
        transaction=transaction.model_dump(),
        decision=final_state.action or "unknown",
        fallback_text=shap_explanation,
    )

    # Response
    return {
        "transaction_id": transaction.transaction_id,
        "fraud_probability": round(fraud_probability, 4),
        "anomaly_score": round(anomaly_score, 4),
        "decision": final_state.action,
        "explanation": explanation_text,
        "feature_contributions": shap_values,
    }
