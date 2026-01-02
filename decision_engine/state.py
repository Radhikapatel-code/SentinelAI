from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DecisionState:
    """
    Represents a decision state for a transaction.
    """

    transaction_id: int
    fraud_probability: float
    anomaly_score: float
    action: Optional[str] = None  # approve, block, review

    def is_terminal(self) -> bool:
        """
        A state is terminal once an action has been chosen.
        """
        return self.action is not None

    def possible_actions(self):
        """
        Returns all possible actions from this state.
        """
        return ["approve", "block", "review"]
