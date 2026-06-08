"""
SentinelAI A* Decision Engine Module.

Implements an A* search algorithm to select the optimal fraud handling
action. Unlike traditional classifiers that output a binary prediction,
this engine evaluates all possible actions and selects the one that
minimizes total expected cost.

The search models each transaction as a state-space problem:
    - Initial state: Transaction with ML-derived risk scores (no action)
    - Actions: approve, block, review
    - Terminal state: A state with an action assigned
    - Cost: Expected cost computed by the CostFunction

Since this is a single-step decision (one action per transaction),
the heuristic h(n) = 0 and the search reduces to selecting the
minimum-cost action. The A* framework is retained for extensibility
to multi-step decision chains (e.g., "flag → verify → block").

Example:
    >>> engine = AStarDecisionEngine(cost_function)
    >>> result = engine.search(initial_state)
    >>> print(result.action)  # "block"
"""

import heapq
from typing import Optional

from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction


class AStarDecisionEngine:
    """Uses A* search to select the lowest-cost decision action.

    The engine maintains a priority queue of states ordered by
    estimated total cost (g + h). For each non-terminal state,
    it expands all possible actions and pushes the resulting states
    onto the queue. The first terminal state popped is optimal.

    Attributes:
        cost_function: The CostFunction used to evaluate actions.
    """

    def __init__(self, cost_function: CostFunction) -> None:
        """Initialize the decision engine with a cost function.

        Args:
            cost_function: The cost function that defines expected
                costs for each action.
        """
        self.cost_function: CostFunction = cost_function

    def search(self, initial_state: DecisionState) -> DecisionState:
        """Perform A* search to find the optimal decision.

        Explores all possible actions from the initial state and returns
        the terminal state with the lowest total cost.

        Args:
            initial_state: The starting state containing transaction
                risk scores but no action assigned.

        Returns:
            DecisionState: A terminal state with the optimal action
                assigned (action is one of "approve", "block", "review").

        Raises:
            RuntimeError: If no valid decision can be found (should
                never occur with a well-formed state).
        """
        # Priority queue: (estimated_total_cost, actual_cost, state)
        open_set: list[tuple[float, float, DecisionState]] = []
        heapq.heappush(open_set, (0.0, 0.0, initial_state))

        best_cost: dict[Optional[str], float] = {}

        while open_set:
            _, current_cost, current_state = heapq.heappop(open_set)

            if current_state.is_terminal():
                return current_state

            for action in current_state.possible_actions():
                next_state = DecisionState(
                    transaction_id=current_state.transaction_id,
                    fraud_probability=current_state.fraud_probability,
                    anomaly_score=current_state.anomaly_score,
                    action=action,
                )

                action_cost: float = self.cost_function.compute(
                    current_state, action
                )

                total_cost: float = current_cost + action_cost

                if (
                    next_state.action not in best_cost
                    or total_cost < best_cost[next_state.action]
                ):
                    best_cost[next_state.action] = total_cost

                    # Heuristic = 0 (single-step decision)
                    estimated_total_cost: float = total_cost

                    heapq.heappush(
                        open_set,
                        (estimated_total_cost, total_cost, next_state),
                    )

        raise RuntimeError("No decision found")
