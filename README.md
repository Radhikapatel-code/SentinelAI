# 🧠 SentinelAI
### An Explainable AI Decision Engine for Intelligent Fraud Detection

![SentinelAI Banner](https://raw.githubusercontent.com/your-username/SentinelAI/main/assets/banner.png)

---

## 📌 Overview

**SentinelAI** is an **Explainable AI-based decision engine** designed to detect and handle fraudulent financial transactions intelligently.

Unlike traditional systems that rely solely on a machine learning classifier, SentinelAI treats fraud detection as a **decision-making problem**, not just a prediction task.

The system combines:
- Machine Learning (Anomaly Detection)
- Rule-Based Reasoning
- Cost-Based AI Search
- Explainable AI (XAI)

to make **optimal, transparent, and auditable decisions**.

> 💡 *Fraud detection is not just about identifying risk — it’s about choosing the right action.*

---

## 🚀 Key Features

- 🔍 Hybrid Intelligence (ML + Rules + Search)
- 🧠 Cost-Aware Decision Making using A* Search
- 📊 Explainable AI outputs (SHAP + Natural Language)
- ⚡ Real-time transaction simulation
- 🧩 Modular and extensible system design

---

## 🧠 System Architecture

Transaction Data
↓
Feature Extraction
↓
Anomaly Detection (Isolation Forest)
↓
Rule-Based Risk Evaluation
↓
AI Decision Engine (A* Search)
↓
Explainability Layer (SHAP + LLM)
↓
Interactive Dashboard


---

## 🧩 Why SentinelAI is Different

### ❌ Traditional Fraud Systems
- Single model prediction
- Black-box decisions
- No cost awareness

### ✅ SentinelAI Approach
- Multi-layer reasoning
- Explicit cost modeling
- Transparent and explainable decisions

SentinelAI explicitly models **false positive vs false negative trade-offs**, which is critical in real-world financial systems.

---

## 🧠 AI Decision Engine (Core Innovation)

Each transaction is modeled as a **state** with the following possible actions:

| Action | Description |
|------|------------|
| Approve | Allow transaction |
| Block | Reject transaction |
| Manual Review | Escalate for human verification |

### 🧮 Cost Function

Total Cost = Fraud Loss + Customer Impact + Operational Cost

An **A\*** search algorithm selects the action that minimizes total expected cost instead of blindly trusting a classifier’s output.

---

## 🔍 Machine Learning Components

### Anomaly Detection
- Isolation Forest
- Learns normal transaction behavior
- Flags statistically rare transactions

### Classification (Optional)
- Logistic Regression / XGBoost
- Produces probabilistic fraud scores

---

## 📖 Explainable AI (XAI)

Each decision includes:
- Risk score
- Top contributing features
- Natural language explanation

**Example Explanation:**

> “Transaction flagged due to unusually high amount (₹92,000), new device usage, and rapid location change.”

This ensures the system is:
- Auditable
- Trustworthy
- Business-ready

---

## 🖥️ Dashboard (Streamlit)

### Dashboard Features
- Live transaction simulation
- Risk meter
- Decision outcome (Approve / Block / Review)
- SHAP feature importance visualization
- Explanation panel

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| Language | Python |
| ML | Scikit-learn |
| Search | A* / Best-First Search |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit |
| Dataset | Credit Card Fraud Dataset (Kaggle) |

---

## 🧪 Dataset

- Credit Card Fraud Detection Dataset
- Highly imbalanced dataset with anonymized features
- Used for anomaly detection and risk modeling

---

## ▶️ How to Run

```bash
# Clone repository
git clone https://github.com/your-username/SentinelAI.git
cd SentinelAI

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn api.main:app --reload

# Run dashboard
streamlit run dashboard/app.py
```
---
## Learning Outcomes

This project demonstrates:

- AI search algorithms applied to real-world problems

- Cost-sensitive decision making

- Explainable AI techniques

- End-to-end system design

- Strong algorithmic thinking and engineering maturity
  
---
## 👤 Author

[Radhika Sanagadhiya]
Undergrad in Information and Communication Technology (ICT) with minors in CS

Interests: AI Systems, Decision Intelligence, Algorithmic Problem Solving

feel free to contact me for any query - 📧 rp773061@gmail.com

---
## ⭐ Final Note

SentinelAI is not just a project — it is a decision-making system.
