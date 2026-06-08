"""
SentinelAI Decision State Module.

Defines the immutable state representation for the A* decision engine.
Each transaction is modeled as a state with possible actions (approve,
block, review), and the engine searches for the optimal action.

The state is a frozen dataclass — once created, it cannot be modified.
This ensures the A* search tree maintains consistent state history.
"""

from dataclasses import dataclass
from typing import Optional


# Valid actions the decision engine can take
VALID_ACTIONS: list[str] = ["approve", "block", "review"]


@dataclass(frozen=True)
class DecisionState:
    """Represents a decision state for a transaction.

    A state captures the ML-derived risk metrics for a transaction
    and optionally the action chosen by the decision engine.

    Attributes:
        transaction_id: Unique identifier for the transaction.
        fraud_probability: Probability of fraud from the classifier (0.0-1.0).
        anomaly_score: Anomaly score from Isolation Forest.
            Lower values indicate more anomalous transactions.
        action: The chosen action, or None if no decision has been made yet.
            Must be one of: "approve", "block", "review".
    """

    transaction_id: int
    fraud_probability: float
    anomaly_score: float
    action: Optional[str] = None

    def is_terminal(self) -> bool:
        """Check if this state is terminal (a decision has been made).

        A state is terminal once an action has been assigned. The A*
        search returns when it reaches a terminal state.

        Returns:
            bool: True if an action has been chosen, False otherwise.
        """
        return self.action is not None

    def possible_actions(self) -> list[str]:
        """Return all possible actions from this state.

        Returns:
            list[str]: The three possible actions:
                - "approve": Allow the transaction
                - "block": Reject the transaction
                - "review": Escalate for manual verification
        """
        return VALID_ACTIONS.copy()
