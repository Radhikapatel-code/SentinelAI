import heapq
from decision_engine.state import DecisionState


class AStarDecisionEngine:
    """
    Uses A* search to select the lowest-cost decision action.
    """

    def __init__(self, cost_function):
        self.cost_function = cost_function

    def search(self, initial_state: DecisionState) -> DecisionState:
        """
        Perform A* search to find the optimal decision.
        """

        # Priority queue: (estimated_total_cost, actual_cost, state)
        open_set = []
        heapq.heappush(open_set, (0, 0, initial_state))

        best_cost = {}

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

                action_cost = self.cost_function.compute(
                    current_state, action
                )

                total_cost = current_cost + action_cost

                if (
                    next_state.action not in best_cost
                    or total_cost < best_cost[next_state.action]
                ):
                    best_cost[next_state.action] = total_cost

                    # Heuristic = 0 (since single-step decision)
                    estimated_total_cost = total_cost

                    heapq.heappush(
                        open_set,
                        (estimated_total_cost, total_cost, next_state),
                    )

        raise RuntimeError("No decision found")
