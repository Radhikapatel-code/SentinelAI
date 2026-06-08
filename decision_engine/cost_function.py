"""
SentinelAI Cost Function Module.

Defines the expected cost model for each possible decision action.
The cost function is the core of SentinelAI's decision intelligence —
it explicitly models the trade-offs between different types of errors:

    - **False Negative** (approving fraud): Cost = transaction amount.
      The organization absorbs the financial loss.
    - **False Positive** (blocking legitimate): Cost = $5-15.
      Customer service call + potential customer churn.
    - **Manual Review**: Cost = $20 per case.
      Analyst time to investigate the transaction.

The A* search engine uses these costs to select the action that
minimizes total expected cost, rather than blindly trusting a
classifier's binary output.

Example:
    >>> cf = CostFunction(fraud_loss=1000, false_positive_cost=50, review_cost=20)
    >>> state = DecisionState(tx_id=1, fraud_probability=0.8, anomaly_score=-0.2)
    >>> cf.compute(state, "approve")  # 0.8 × 1000 = 800.0
    >>> cf.compute(state, "block")    # 0.2 × 50  = 10.0
    >>> cf.compute(state, "review")   # 20.0
"""

from typing import Any


class CostFunction:
    """Computes expected cost for each decision action.

    The cost model uses fraud probability to calculate the expected
    cost of each action, accounting for uncertainty in the prediction:

        approve: E[cost] = P(fraud) × fraud_loss
        block:   E[cost] = P(legit) × false_positive_cost
        review:  E[cost] = review_cost (fixed)

    Decision boundaries (with defaults fraud_loss=1000, FP=50, review=20):
        - P(fraud) < 0.02  → approve is cheapest
        - 0.02 ≤ P(fraud) < 0.60 → review is cheapest
        - P(fraud) ≥ 0.60 → block is cheapest

    Attributes:
        fraud_loss: Cost of allowing a fraudulent transaction.
        false_positive_cost: Cost of blocking a legitimate transaction.
        review_cost: Fixed cost per manual review case.
    """

    def __init__(
        self,
        fraud_loss: float = 1000.0,
        false_positive_cost: float = 50.0,
        review_cost: float = 20.0,
    ) -> None:
        """Initialize the cost function with configurable weights.

        Args:
            fraud_loss: Expected loss when a fraudulent transaction is
                approved. In production, this typically equals the
                transaction amount. Default: 1000.0.
            false_positive_cost: Cost of incorrectly blocking a legitimate
                transaction. Includes customer service overhead ($5-10)
                and customer churn risk ($5-15). Default: 50.0.
            review_cost: Fixed cost per manual review case, representing
                analyst time (~$20 per case). Default: 20.0.
        """
        self.fraud_loss: float = fraud_loss
        self.false_positive_cost: float = false_positive_cost
        self.review_cost: float = review_cost

    def compute(self, state: Any, action: str) -> float:
        """Compute expected cost for taking an action in a given state.

        Args:
            state: A DecisionState object with a fraud_probability attribute.
            action: The action to evaluate. Must be one of:
                "approve", "block", or "review".

        Returns:
            float: The expected cost of taking this action.

        Raises:
            ValueError: If the action is not recognized.
        """
        p_fraud: float = state.fraud_probability

        if action == "approve":
            # Risk: fraud goes through → organization absorbs the loss
            return p_fraud * self.fraud_loss

        elif action == "block":
            # Risk: legitimate transaction blocked → customer impact
            return (1 - p_fraud) * self.false_positive_cost

        elif action == "review":
            # Fixed operational cost regardless of fraud probability
            return self.review_cost

        else:
            raise ValueError(f"Unknown action: {action}")

    def describe(self) -> str:
        """Return a human-readable description of the cost configuration.

        Useful for logging and the demo script. Shows current weights
        and the resulting decision boundaries.

        Returns:
            str: Multi-line description of the cost model.
        """
        # Calculate decision boundaries
        # approve < review when: p × fraud_loss < review_cost
        approve_review_boundary = self.review_cost / self.fraud_loss
        # block < review when: (1-p) × FP_cost < review_cost
        block_review_boundary = 1 - (self.review_cost / self.false_positive_cost)

        return (
            f"Cost Configuration:\n"
            f"  Fraud Loss (false negative):     ${self.fraud_loss:,.0f}\n"
            f"  False Positive Cost:              ${self.false_positive_cost:,.0f}\n"
            f"  Manual Review Cost:               ${self.review_cost:,.0f}\n"
            f"\n"
            f"Decision Boundaries:\n"
            f"  P(fraud) < {approve_review_boundary:.3f}  → APPROVE\n"
            f"  {approve_review_boundary:.3f} ≤ P(fraud) < {block_review_boundary:.3f}  → REVIEW\n"
            f"  P(fraud) ≥ {block_review_boundary:.3f}  → BLOCK"
        )
