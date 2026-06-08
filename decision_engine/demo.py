"""
SentinelAI Decision Engine Demo.

A standalone script that demonstrates the full fraud detection pipeline:
    Anomaly Score → Fraud Probability → A* Search → Decision

Shows the search expanding nodes, computing costs, and selecting the
minimum-cost action with verbose step-by-step output.

Usage:
    python -m decision_engine.demo

    With custom parameters:
    python -m decision_engine.demo --amount 89000 --device-change 1 --merchant-risk 0.95
"""

import argparse
import sys
import os
import time
from typing import Optional

# Ensure UTF-8 output for Windows terminals (prevents UnicodeEncodeError)
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.anomaly_detector import AnomalyDetector
from models.classifier import FraudClassifier
from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine
from explainability.shap_explainer import ShapExplainer


# ──────────────────────────────────────────────
# ANSI Colors for Terminal Output
# ──────────────────────────────────────────────
class _Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


C = _Colors()


# ──────────────────────────────────────────────
# Predefined Demo Transactions
# ──────────────────────────────────────────────
DEMO_TRANSACTIONS: dict[str, dict] = {
    "legitimate": {
        "transaction_id": 1001,
        "amount": 1200,
        "transaction_time": 14,
        "location_change": 0,
        "device_change": 0,
        "merchant_risk": 0.05,
    },
    "fraudulent": {
        "transaction_id": 1002,
        "amount": 89000,
        "transaction_time": 2,
        "location_change": 1,
        "device_change": 1,
        "merchant_risk": 0.95,
    },
    "borderline": {
        "transaction_id": 1003,
        "amount": 15000,
        "transaction_time": 22,
        "location_change": 1,
        "device_change": 0,
        "merchant_risk": 0.45,
    },
}


def print_banner() -> None:
    """Print the demo banner."""
    print(f"\n{C.BOLD}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  🧠 SentinelAI — Decision Engine Demo{C.RESET}")
    print(f"{C.BOLD}{'=' * 60}{C.RESET}")


def print_transaction(tx: dict) -> None:
    """Print transaction details in a formatted box.

    Args:
        tx: Transaction dictionary.
    """
    print(f"\n{C.BOLD}📋 Transaction Details:{C.RESET}")
    print(f"  {'─' * 40}")
    for key, value in tx.items():
        if key == "transaction_id":
            print(f"  │ {C.DIM}ID:{C.RESET}              {value}")
        elif key == "amount":
            print(f"  │ {C.BOLD}Amount:{C.RESET}          ₹{value:,.0f}")
        elif key == "transaction_time":
            print(f"  │ Time:{' ' * 11}{value}:00")
        elif key == "location_change":
            flag = f"{C.RED}YES{C.RESET}" if value else f"{C.GREEN}NO{C.RESET}"
            print(f"  │ Location Change: {flag}")
        elif key == "device_change":
            flag = f"{C.RED}YES{C.RESET}" if value else f"{C.GREEN}NO{C.RESET}"
            print(f"  │ Device Change:   {flag}")
        elif key == "merchant_risk":
            color = C.RED if value > 0.5 else (C.YELLOW if value > 0.2 else C.GREEN)
            print(f"  │ Merchant Risk:   {color}{value:.2f}{C.RESET}")
    print(f"  {'─' * 40}")


def run_pipeline(
    tx: dict,
    anomaly_detector: AnomalyDetector,
    classifier: FraudClassifier,
    cost_function: CostFunction,
    explainer: ShapExplainer,
) -> None:
    """Run the full decision pipeline with verbose output.

    Args:
        tx: Transaction dictionary.
        anomaly_detector: Trained anomaly detector.
        classifier: Trained fraud classifier.
        cost_function: Cost function for the decision engine.
        explainer: SHAP explainer instance.
    """
    tx_df = pd.DataFrame([tx])

    # ── Step 1: Anomaly Detection ──
    print(f"\n{C.BOLD}{C.BLUE}Step 1: Anomaly Detection (Isolation Forest){C.RESET}")
    start = time.time()
    anomaly_score = anomaly_detector.anomaly_score(tx_df)[0]
    prediction = anomaly_detector.predict(tx_df)[0]
    elapsed_ms = (time.time() - start) * 1000

    status = f"{C.RED}ANOMALOUS{C.RESET}" if prediction == -1 else f"{C.GREEN}NORMAL{C.RESET}"
    print(f"  Anomaly Score: {anomaly_score:.4f}")
    print(f"  Status:        {status}")
    print(f"  {C.DIM}Time: {elapsed_ms:.2f}ms{C.RESET}")

    # ── Step 2: Fraud Probability ──
    print(f"\n{C.BOLD}{C.BLUE}Step 2: Fraud Probability (Logistic Regression){C.RESET}")
    start = time.time()
    fraud_probability = classifier.predict_proba(tx_df)[0]
    elapsed_ms = (time.time() - start) * 1000

    color = C.RED if fraud_probability > 0.7 else (C.YELLOW if fraud_probability > 0.3 else C.GREEN)
    print(f"  P(fraud): {color}{fraud_probability:.4f}{C.RESET} ({fraud_probability * 100:.1f}%)")
    print(f"  {C.DIM}Time: {elapsed_ms:.2f}ms{C.RESET}")

    # ── Step 3: A* Decision Search ──
    print(f"\n{C.BOLD}{C.BLUE}Step 3: A* Decision Search{C.RESET}")
    print(f"  {C.DIM}Cost weights: fraud_loss={cost_function.fraud_loss}, "
          f"FP_cost={cost_function.false_positive_cost}, "
          f"review={cost_function.review_cost}{C.RESET}")
    print()

    state = DecisionState(
        transaction_id=tx["transaction_id"],
        fraud_probability=fraud_probability,
        anomaly_score=anomaly_score,
    )

    # Manually compute costs for each action to show the search
    costs: dict[str, float] = {}
    for action in state.possible_actions():
        cost = cost_function.compute(state, action)
        costs[action] = cost

        # Show the formula
        if action == "approve":
            formula = f"{fraud_probability:.4f} × {cost_function.fraud_loss}"
        elif action == "block":
            formula = f"{1 - fraud_probability:.4f} × {cost_function.false_positive_cost}"
        else:
            formula = f"fixed cost"

        print(f"  → {action.upper():>7s}: cost = {formula} = {C.BOLD}{cost:.2f}{C.RESET}")

    # Find optimal
    start = time.time()
    engine = AStarDecisionEngine(cost_function)
    result = engine.search(state)
    elapsed_ms = (time.time() - start) * 1000

    decision_color = {
        "approve": C.GREEN,
        "block": C.RED,
        "review": C.YELLOW,
    }
    decision_emoji = {
        "approve": "🟢",
        "block": "🔴",
        "review": "🟡",
    }

    dc = decision_color.get(result.action, "")
    de = decision_emoji.get(result.action, "")
    print(f"\n  {C.BOLD}✓ Optimal action: {dc}{de} {result.action.upper()}{C.RESET}"
          f" (cost: {costs[result.action]:.2f})")
    print(f"  {C.DIM}Search time: {elapsed_ms:.2f}ms{C.RESET}")

    # ── Step 4: Explainability ──
    print(f"\n{C.BOLD}{C.BLUE}Step 4: Explainability (SHAP){C.RESET}")
    shap_values = explainer.explain(tx_df)
    explanation = explainer.generate_text_explanation(shap_values, tx_df)

    # Show feature contributions
    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, value in sorted_features:
        direction = "↑" if value > 0 else "↓"
        color = C.RED if value > 0 else C.GREEN
        bar_len = int(abs(value) * 50)
        bar = "█" * min(bar_len, 30)
        print(f"  {feature:>20s}: {color}{direction} {value:+.4f}{C.RESET} {C.DIM}{bar}{C.RESET}")

    print(f"\n  {C.BOLD}💬 Explanation:{C.RESET}")
    print(f"  \"{explanation}\"")


def main() -> None:
    """Run the demo with predefined or custom transactions."""
    parser = argparse.ArgumentParser(
        description="SentinelAI Decision Engine Demo"
    )
    parser.add_argument("--amount", type=float, default=None, help="Transaction amount")
    parser.add_argument("--time", type=int, default=14, help="Transaction hour (0-23)")
    parser.add_argument("--location-change", type=int, default=0, choices=[0, 1])
    parser.add_argument("--device-change", type=int, default=0, choices=[0, 1])
    parser.add_argument("--merchant-risk", type=float, default=0.2)
    parser.add_argument(
        "--fraud-loss", type=float, default=1000,
        help="Cost of allowing fraud through (default: 1000)"
    )
    parser.add_argument(
        "--fp-cost", type=float, default=50,
        help="Cost of blocking legitimate transaction (default: 50)"
    )
    parser.add_argument(
        "--review-cost", type=float, default=20,
        help="Cost of manual review (default: 20)"
    )
    args = parser.parse_args()

    print_banner()

    # Load training data and train models
    print(f"\n{C.DIM}Loading models...{C.RESET}")
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "transactions.csv"
    )
    df = pd.read_csv(data_path)

    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(df)

    classifier_model = FraudClassifier()
    classifier_model.fit(df)

    cost_fn = CostFunction(
        fraud_loss=args.fraud_loss,
        false_positive_cost=args.fp_cost,
        review_cost=args.review_cost,
    )

    explainer = ShapExplainer(classifier_model.pipeline)
    print(f"{C.GREEN}✓ Models loaded{C.RESET}")

    if args.amount is not None:
        # Custom transaction from CLI args
        custom_tx = {
            "transaction_id": 9999,
            "amount": args.amount,
            "transaction_time": args.time,
            "location_change": args.location_change,
            "device_change": args.device_change,
            "merchant_risk": args.merchant_risk,
        }
        print_transaction(custom_tx)
        run_pipeline(custom_tx, anomaly_detector, classifier_model, cost_fn, explainer)
    else:
        # Run all demo transactions
        for name, tx in DEMO_TRANSACTIONS.items():
            print(f"\n{'═' * 60}")
            print(f"{C.BOLD}{C.HEADER}  📌 Demo: {name.upper()} Transaction{C.RESET}")
            print(f"{'═' * 60}")
            print_transaction(tx)
            run_pipeline(tx, anomaly_detector, classifier_model, cost_fn, explainer)

    print(f"\n{'═' * 60}")
    print(f"{C.BOLD}{C.GREEN}  ✅ Demo complete{C.RESET}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
