class CostFunction:
    """
    Defines cost associated with each decision action.
    """

    def __init__(
        self,
        fraud_loss=1000,
        false_positive_cost=50,
        review_cost=20,
    ):
        """
        Parameters:
        fraud_loss: cost of allowing a fraudulent transaction
        false_positive_cost: cost of blocking a legitimate transaction
        review_cost: cost of manual review
        """
        self.fraud_loss = fraud_loss
        self.false_positive_cost = false_positive_cost
        self.review_cost = review_cost

    def compute(self, state, action: str) -> float:
        """
        Compute expected cost for taking an action in a given state.
        """

        p_fraud = state.fraud_probability

        if action == "approve":
            # Cost if fraud actually happens
            return p_fraud * self.fraud_loss

        elif action == "block":
            # Cost if transaction was actually legitimate
            return (1 - p_fraud) * self.false_positive_cost

        elif action == "review":
            # Fixed operational cost
            return self.review_cost

        else:
            raise ValueError(f"Unknown action: {action}")
