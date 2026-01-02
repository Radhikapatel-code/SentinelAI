import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier
from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine
from explainability.shap_explainer import ShapExplainer

# ------------------ #
# Initialize App
# ------------------ #
app = FastAPI(title="SentinelAI API")

# ------------------ #
# Load Data
# ------------------ #
DATA_PATH = "data/transactions.csv"
df = pd.read_csv(DATA_PATH)

# ------------------ #
# Train Models (once at startup)
# ------------------ #
anomaly_detector = AnomalyDetector()
anomaly_detector.fit(df)

classifier = FraudClassifier()
classifier.fit(df)

cost_function = CostFunction()
decision_engine = AStarDecisionEngine(cost_function)

explainer = ShapExplainer(classifier.pipeline)

# ------------------ #
# Request Schema
# ------------------ #
class TransactionInput(BaseModel):
    transaction_id: int
    amount: float
    transaction_time: int
    location_change: int
    device_change: int
    merchant_risk: float


# ------------------ #
# API Endpoint
# ------------------ #
@app.post("/analyze")
def analyze_transaction(transaction: TransactionInput):
    # Convert input to DataFrame
    tx_df = pd.DataFrame([transaction.dict()])

    # ML Predictions
    fraud_probability = classifier.predict_proba(tx_df)[0]
    anomaly_score = anomaly_detector.anomaly_score(tx_df)[0]

    # Decision State
    state = DecisionState(
        transaction_id=transaction.transaction_id,
        fraud_probability=fraud_probability,
        anomaly_score=anomaly_score,
    )

    # Run Decision Engine
    final_state = decision_engine.search(state)

    # Explainability
    shap_values = explainer.explain(tx_df)
    explanation_text = explainer.generate_text_explanation(shap_values)

    # Response
    return {
        "transaction_id": transaction.transaction_id,
        "fraud_probability": round(float(fraud_probability), 4),
        "anomaly_score": round(float(anomaly_score), 4),
        "decision": final_state.action,
        "explanation": explanation_text,
        "feature_contributions": shap_values,
    }
