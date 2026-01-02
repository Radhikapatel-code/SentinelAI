import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(
    page_title="SentinelAI Dashboard",
    layout="centered",
)

st.title("🧠 SentinelAI")
st.subheader("Explainable AI Decision Engine for Fraud Detection")

st.markdown("---")

st.header("🔢 Transaction Input")

transaction_id = st.number_input("Transaction ID", value=1, step=1)
amount = st.number_input("Amount", value=5000.0, step=100.0)
transaction_time = st.slider("Transaction Time (Hour)", 0, 23, 12)
location_change = st.selectbox("Location Change", [0, 1])
device_change = st.selectbox("Device Change", [0, 1])
merchant_risk = st.slider("Merchant Risk Score", 0.0, 1.0, 0.2)

if st.button("Analyze Transaction"):
    payload = {
        "transaction_id": transaction_id,
        "amount": amount,
        "transaction_time": transaction_time,
        "location_change": location_change,
        "device_change": device_change,
        "merchant_risk": merchant_risk,
    }

    with st.spinner("Analyzing transaction..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.markdown("---")
        st.header("📊 Analysis Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Fraud Probability",
            f"{result['fraud_probability'] * 100:.2f}%",
        )

        col2.metric(
            "Anomaly Score",
            f"{result['anomaly_score']:.3f}",
        )

        decision_color = {
            "approve": "🟢 APPROVE",
            "review": "🟡 REVIEW",
            "block": "🔴 BLOCK",
        }

        col3.metric(
            "Decision",
            decision_color.get(result["decision"], result["decision"]),
        )

        st.markdown("---")
        st.subheader("🧠 Explanation")

        st.write(result["explanation"])

        st.markdown("---")
        st.subheader("📌 Feature Contributions")

        contrib_df = pd.DataFrame(
            result["feature_contributions"].items(),
            columns=["Feature", "Contribution"],
        ).sort_values(
            by="Contribution", key=abs, ascending=False
        )

        st.bar_chart(contrib_df.set_index("Feature"))

    else:
        st.error("Failed to analyze transaction. Is the API running?")
