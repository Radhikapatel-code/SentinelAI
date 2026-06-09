"""
SentinelAI Interactive Dashboard.

Streamlit-based frontend for real-time fraud detection analysis.
Communicates with the FastAPI backend via HTTP — never imports
directly from the ML models or decision engine.

Usage:
    streamlit run dashboard/app.py

Environment Variables:
    API_URL: Backend API URL (default: http://127.0.0.1:8000)
"""

import os
import time

import streamlit as st
import requests
import pandas as pd

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
API_URL: str = os.environ.get("API_URL", "http://127.0.0.1:8000")
ANALYZE_ENDPOINT: str = f"{API_URL}/analyze"

# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI Dashboard",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 SentinelAI")
st.subheader("Explainable AI Decision Engine for Fraud Detection")

st.markdown("---")

# ──────────────────────────────────────────────
# Transaction Input Form
# ──────────────────────────────────────────────
st.header("🔢 Transaction Input")

transaction_id: int = st.number_input(
    "Transaction ID", value=1, step=1, min_value=1
)
amount: float = st.number_input(
    "Amount (₹)", value=5000.0, step=100.0, min_value=0.01
)
transaction_time: int = st.slider("Transaction Time (Hour)", 0, 23, 12)
location_change: int = st.selectbox("Location Change", [0, 1])
device_change: int = st.selectbox("Device Change", [0, 1])
merchant_risk: float = st.slider(
    "Merchant Risk Score", 0.0, 1.0, 0.2, step=0.05
)

# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────
if st.button("🔍 Analyze Transaction", type="primary"):
    payload: dict = {
        "transaction_id": transaction_id,
        "amount": amount,
        "transaction_time": transaction_time,
        "location_change": location_change,
        "device_change": device_change,
        "merchant_risk": merchant_risk,
    }

    with st.spinner("Analyzing transaction... (first request may take ~30s if the server is waking up)"):
        response = None
        for attempt in range(2):
            try:
                response = requests.post(ANALYZE_ENDPOINT, json=payload, timeout=90)
                break
            except requests.exceptions.ReadTimeout:
                if attempt == 0:
                    st.warning("⏳ Server is waking up (free tier cold start). Retrying...")
                    time.sleep(5)
                else:
                    st.error(
                        "❌ Request timed out. The backend may still be starting up. "
                        "Please wait a moment and try again."
                    )
                    st.stop()
            except requests.ConnectionError:
                st.error(
                    "❌ Cannot connect to the API. "
                    f"Is the backend running at {API_URL}?"
                )
                st.info("Start the backend with: `uvicorn api.main:app --reload`")
                st.stop()

    if response.status_code == 200:
        result: dict = response.json()

        st.markdown("---")
        st.header("📊 Analysis Result")

        # ── Metric Cards ──
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.2f}%",
        )

        col2.metric(
            "Anomaly Score",
            f"{result['anomaly_score']:.3f}",
        )

        decision_display: dict[str, str] = {
            "approve": "🟢 APPROVE",
            "review": "🟡 REVIEW",
            "block": "🔴 BLOCK",
        }

        col3.metric(
            "Decision",
            decision_display.get(result["decision"], result["decision"]),
        )

        # ── Risk Level Bar ──
        st.markdown("---")
        st.subheader("📈 Risk Level")
        risk_pct: float = result["fraud_probability"]
        st.progress(min(risk_pct, 1.0))

        if risk_pct < 0.3:
            st.success("Low Risk — Transaction appears legitimate")
        elif risk_pct < 0.7:
            st.warning("Medium Risk — Transaction requires review")
        else:
            st.error("High Risk — Transaction flagged as potentially fraudulent")

        # ── Explanation ──
        st.markdown("---")
        st.subheader("🧠 Explanation")
        st.info(result["explanation"])

        # ── Feature Contributions ──
        st.markdown("---")
        st.subheader("📌 Feature Contributions (SHAP)")

        contrib_df: pd.DataFrame = pd.DataFrame(
            result["feature_contributions"].items(),
            columns=["Feature", "Contribution"],
        ).sort_values(by="Contribution", key=abs, ascending=False)

        st.bar_chart(contrib_df.set_index("Feature"))

    else:
        st.error(
            f"❌ API returned status {response.status_code}. "
            "Check the backend logs for details."
        )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.caption(
    "SentinelAI v0.2.0 | "
    f"[API Docs]({API_URL}/docs) | "
    "Built with FastAPI + Streamlit"
)
