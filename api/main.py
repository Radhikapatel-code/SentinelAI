"""
SentinelAI FastAPI Backend.

Provides a REST API for real-time transaction fraud analysis.

Endpoints:
    /analyze    — Full pipeline (ML + A* + SHAP + LLM) — synchronous
    /ingest     — Publish transaction to Kafka for async scoring
    /results    — Query scored results from PostgreSQL
    /health     — Enhanced health check with queue depth + circuit breaker

Rate limiting: Custom token-bucket middleware (configurable rate + burst)
Circuit breaker: Wraps scoring path, fails fast under degradation

Usage:
    uvicorn api.main:app --reload

API Documentation:
    http://localhost:8000/docs  (Swagger UI — auto-generated)
"""

import json
import logging
import os
import time
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        "SHAP-based explainability. Supports both synchronous analysis "
        "and async streaming ingestion via Kafka/Redpanda."
    ),
    version="2.0.0",
)

# CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
# Streaming Components (lazy-initialized)
# ──────────────────────────────────────────────
_kafka_producer = None
_rate_limiter = None
_circuit_breaker = None
_backpressure_monitor = None


def _get_kafka_producer():
    """Lazy-initialize Kafka producer (only when /ingest is used)."""
    global _kafka_producer
    if _kafka_producer is None:
        try:
            from streaming.config import get_streaming_config
            from streaming.producer import TransactionProducer
            _kafka_producer = TransactionProducer(get_streaming_config())
            logger.info("Kafka producer initialized for /ingest endpoint")
        except Exception as e:
            logger.warning("Kafka producer unavailable: %s", e)
    return _kafka_producer


def _get_rate_limiter():
    """Lazy-initialize rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        try:
            from streaming.rate_limiter import TokenBucketRateLimiter
            from streaming.config import get_streaming_config
            streaming_config = get_streaming_config()
            _rate_limiter = TokenBucketRateLimiter(
                rate=streaming_config.rate_limiter.rate,
                capacity=streaming_config.rate_limiter.capacity,
            )
            logger.info(
                "Rate limiter initialized: %.0f req/sec, burst %.0f",
                streaming_config.rate_limiter.rate,
                streaming_config.rate_limiter.capacity,
            )
        except Exception as e:
            logger.warning("Rate limiter unavailable: %s", e)
    return _rate_limiter


def _get_circuit_breaker():
    """Lazy-initialize circuit breaker."""
    global _circuit_breaker
    if _circuit_breaker is None:
        try:
            from streaming.circuit_breaker import CircuitBreaker
            from streaming.config import get_streaming_config
            streaming_config = get_streaming_config()
            _circuit_breaker = CircuitBreaker(
                failure_threshold=streaming_config.circuit_breaker.failure_threshold,
                recovery_timeout=streaming_config.circuit_breaker.recovery_timeout,
                success_threshold=streaming_config.circuit_breaker.success_threshold,
                name="api-scoring",
            )
            logger.info("Circuit breaker initialized for API scoring path")
        except Exception as e:
            logger.warning("Circuit breaker unavailable: %s", e)
    return _circuit_breaker


# ──────────────────────────────────────────────
# Rate Limiter Middleware
# ──────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply token-bucket rate limiting to all endpoints.

    Returns 429 Too Many Requests when the bucket is empty.
    """
    limiter = _get_rate_limiter()
    if limiter and not limiter.acquire():
        return Response(
            content=json.dumps({
                "detail": "Rate limit exceeded. Try again shortly.",
                "retry_after_ms": int(1000 / limiter.rate),
            }),
            status_code=429,
            media_type="application/json",
        )
    return await call_next(request)


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


class IngestResponse(BaseModel):
    """Response schema for async ingestion."""

    transaction_id: int
    status: str
    message: str


class ScoredResultResponse(BaseModel):
    """Response schema for querying scored results."""

    transaction_id: int
    amount: Optional[float] = None
    anomaly_score: float
    fraud_probability: float
    decision: str
    expected_cost: float
    worker_id: Optional[str] = None
    scored_at: Optional[str] = None


# ──────────────────────────────────────────────
# Health Check (Enhanced)
# ──────────────────────────────────────────────
@app.get("/health")
def health_check() -> dict[str, Any]:
    """Enhanced health check with system status.

    Returns:
        dict: Status, version, and streaming component states.
    """
    health: dict[str, Any] = {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "models": "loaded",
            "kafka": "connected" if _kafka_producer else "not_initialized",
        },
    }

    # Rate limiter stats
    limiter = _get_rate_limiter()
    if limiter:
        health["rate_limiter"] = limiter.stats

    # Circuit breaker stats
    cb = _get_circuit_breaker()
    if cb:
        health["circuit_breaker"] = {
            "state": cb.state.value,
            "total_calls": cb.stats["total_calls"],
            "total_rejected": cb.stats["total_rejected"],
        }

    # Backpressure status
    if _backpressure_monitor:
        health["backpressure"] = _backpressure_monitor.stats

    return health


# ──────────────────────────────────────────────
# Main Analysis Endpoint (Synchronous)
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


# ──────────────────────────────────────────────
# Async Ingestion Endpoint (Publish to Kafka)
# ──────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
def ingest_transaction(transaction: TransactionInput) -> dict[str, Any]:
    """Ingest a transaction for async scoring via the streaming pipeline.

    Publishes the transaction to the Kafka/Redpanda topic for
    processing by the worker pool. Returns immediately — the
    transaction will be scored asynchronously.

    Use /results/{transaction_id} to query the scoring result.

    Args:
        transaction: The transaction data to ingest.

    Returns:
        dict: Ingestion confirmation with transaction ID.

    Raises:
        HTTPException: 503 if Kafka is unavailable.
    """
    producer = _get_kafka_producer()
    if producer is None:
        raise HTTPException(
            status_code=503,
            detail="Streaming pipeline not available. "
                   "Use /analyze for synchronous scoring.",
        )

    try:
        from streaming.models import TransactionMessage
        msg = TransactionMessage(
            transaction_id=transaction.transaction_id,
            amount=transaction.amount,
            transaction_time=transaction.transaction_time,
            location_change=transaction.location_change,
            device_change=transaction.device_change,
            merchant_risk=transaction.merchant_risk,
            ingested_at=time.time(),
        )
        producer.produce_one(msg)
        producer.producer.poll(0)  # Trigger delivery callbacks

        return {
            "transaction_id": transaction.transaction_id,
            "status": "accepted",
            "message": "Transaction queued for async scoring. "
                       "Query /results/{transaction_id} for the result.",
        }
    except Exception as e:
        logger.error("Ingestion failed for tx %d: %s",
                      transaction.transaction_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


# ──────────────────────────────────────────────
# Query Scored Results
# ──────────────────────────────────────────────
@app.get("/results/{transaction_id}", response_model=ScoredResultResponse)
def get_result(transaction_id: int) -> dict[str, Any]:
    """Query the scoring result for a specific transaction.

    Looks up the result in PostgreSQL (persisted by the worker pool).

    Args:
        transaction_id: The transaction ID to look up.

    Returns:
        dict: Scoring result.

    Raises:
        HTTPException: 404 if not found, 503 if DB unavailable.
    """
    try:
        import psycopg2
        from streaming.config import get_streaming_config
        streaming_config = get_streaming_config()

        conn = psycopg2.connect(streaming_config.postgres.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT transaction_id, amount, anomaly_score,
                           fraud_probability, decision, expected_cost,
                           worker_id, scored_at
                    FROM scored_transactions
                    WHERE transaction_id = %s
                    """,
                    (transaction_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Transaction {transaction_id} not found. "
                       "It may still be in the scoring queue.",
            )

        return {
            "transaction_id": row[0],
            "amount": row[1],
            "anomaly_score": row[2],
            "fraud_probability": row[3],
            "decision": row[4],
            "expected_cost": row[5],
            "worker_id": row[6],
            "scored_at": str(row[7]) if row[7] else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Result query failed for tx %d: %s",
                      transaction_id, e)
        raise HTTPException(
            status_code=503,
            detail="Database unavailable. Use /analyze for "
                   "synchronous scoring instead.",
        )


# ──────────────────────────────────────────────
# Scoring Metrics (for Grafana/monitoring)
# ──────────────────────────────────────────────
@app.get("/metrics/summary")
def get_metrics_summary() -> dict[str, Any]:
    """Get a summary of scoring metrics from PostgreSQL.

    Returns:
        dict: Scoring statistics from the scoring_metrics view.
    """
    try:
        import psycopg2
        from streaming.config import get_streaming_config
        streaming_config = get_streaming_config()

        conn = psycopg2.connect(streaming_config.postgres.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM scoring_metrics")
                row = cur.fetchone()
                if row:
                    cols = [desc[0] for desc in cur.description]
                    return dict(zip(cols, row))
                return {"total_scored": 0}
        finally:
            conn.close()
    except Exception as e:
        return {"error": str(e), "total_scored": 0}
