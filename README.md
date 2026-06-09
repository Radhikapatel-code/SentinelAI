# 🧠 SentinelAI
### An Explainable AI Decision Engine for Intelligent Fraud Detection

[![CI](https://github.com/Radhikapatel-code/SentinelAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Radhikapatel-code/SentinelAI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[📚 API Docs](https://sentinelai-uzhn.onrender.com/docs)

![SentinelAI Banner](https://raw.githubusercontent.com/Radhikapatel-code/SentinelAI/main/assets/banner.png)

---

## 📌 Overview

**SentinelAI** is an **Explainable AI-based decision engine** designed to detect and handle fraudulent financial transactions intelligently.

Unlike traditional systems that rely solely on a machine learning classifier, SentinelAI treats fraud detection as a **decision-making problem**, not just a prediction task.

The system combines:
- Machine Learning (Anomaly Detection)
- Rule-Based Reasoning
- Cost-Based AI Search
- Explainable AI (SHAP + LLM)

to make **optimal, transparent, and auditable decisions**.

> 💡 *Fraud detection is not just about identifying risk — it’s about choosing the right action.*

---

## 🚀 Key Features

- 🔍 **Hybrid Intelligence**: Combines ML anomaly scoring with Logistic Regression.
- 🧠 **Cost-Aware Decision Making**: Uses an A* search algorithm to select the action minimizing total expected business cost.
- 📊 **Explainable AI**: Provides feature attribution via SHAP and natural language explanations powered by LLMs (OpenAI).
- ⚡ **Real-Time Simulation**: Interactive Streamlit dashboard to simulate transactions and visualize decisions.
- 🧩 **Production-Ready**: FastAPI backend, fully typed, centralized config, CI/CD pipeline, and 60%+ test coverage.

---

## 🧠 System Architecture

```text
Transaction Data
       ↓
Feature Extraction
       ↓
Anomaly Detection (Isolation Forest) & Fraud Classifier (Logistic Regression)
       ↓
Rule-Based Risk Evaluation
       ↓
AI Decision Engine (A* Search)
       ↓
Explainability Layer (SHAP + LLM)
       ↓
Interactive Dashboard
```

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

## 📊 Results & Performance

Evaluated on the Kaggle Credit Card Fraud Detection dataset.

| Metric | Isolation Forest | Logistic Regression |
|--------|-----------------|-------------------|
| **AUC-ROC** | 0.9474 | 0.9699 |
| **Avg Precision (PR-AUC)** | 0.1781 | 0.7017 |
| **F1 Score** | — | 0.1193 |
| **Precision @90% Recall** | — | 0.0226 |

| Operational Metric | Value |
|--------|-------|
| **Dataset Size** | 284,807 transactions |
| **Fraud Rate** | 0.173% |
| **Avg Decision Time** | ~0.6ms |

---

## 🧠 AI Decision Engine (Core Innovation)

Each transaction is modeled as a **state** with the following possible actions:

| Action | Description |
|------|------------|
| **Approve** | Allow transaction |
| **Block** | Reject transaction |
| **Manual Review** | Escalate for human verification |

### 🧮 Cost Function

The system calculates expected costs based on configured weights (see `config.yaml`):

- **Fraud Loss** (False Negative): Transaction amount (Default: $1000)
- **False Positive Cost**: Customer impact (Default: $50)
- **Review Cost**: Operational overhead (Default: $20)

An **A\*** search algorithm selects the action that minimizes total expected cost instead of blindly trusting a classifier’s output.

---

## 📖 Explainable AI (XAI)

Each decision includes:
- Risk score
- Top contributing features (SHAP values)
- Natural language explanation (LLM-powered)

**Example Explanation:**

> “Transaction blocked — unusually high transaction amount (₹89,000) increased risk, combined with a new device being used and rapid location change.”

This ensures the system is auditable, trustworthy, and business-ready.

---

## ▶️ Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Radhikapatel-code/SentinelAI.git
cd SentinelAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file:
```bash
cp .env.example .env
```
*(Optional)* Add your `OPENAI_API_KEY` to `.env` for LLM-powered explanations.

### 3. Running the System

Start the FastAPI backend:
```bash
uvicorn api.main:app --reload
```
View the auto-generated API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

In a new terminal, start the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

### 4. Running the Demo Script

See the A* decision engine in action in the terminal:
```bash
python -m decision_engine.demo
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-----|-----------|
| **Language** | Python 3.10+ |
| **ML** | Scikit-learn, Pandas, Numpy |
| **Search** | A* / Best-First Search |
| **Explainability** | SHAP, OpenAI API |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Testing/CI** | Pytest, GitHub Actions |

---
## 📸 Demo
### Dashboard
<img width="1911" height="923" alt="image" src="https://github.com/user-attachments/assets/7895b498-20c5-4cb7-84ba-6e72961f21e7" />
### Decision Output
<img width="1913" height="699" alt="image" src="https://github.com/user-attachments/assets/6147be66-5772-4ada-a5f1-637a029fe8d2" />
<img width="1904" height="949" alt="image" src="https://github.com/user-attachments/assets/79f8c529-5e97-4f4b-8b06-821212a0d525" />


## 🧪 Development & Testing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.

Run tests with coverage:
```bash
pytest --cov=. --cov-report=term-missing
```

---

## 👤 Author

**Radhika Sanagadhiya**  
Undergrad in Information and Communication Technology (ICT) with minors in CS

Interests: AI Systems, Decision Intelligence, Algorithmic Problem Solving  
Contact: 📧 rp773061@gmail.com

---

## ⭐ Final Note

SentinelAI is not just a project — it is a decision-making system.
