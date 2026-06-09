"""
Tests for the Decision Engine module.

Validates:
- A* search returns correct actions for different risk levels
- CostFunction computes expected costs correctly
- DecisionState terminal/non-terminal behavior
- Cost function decision boundaries match documented math
"""

import pytest

from decision_engine.state import DecisionState, VALID_ACTIONS
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine


class TestDecisionState:
    """Tests for DecisionState dataclass."""

    def test_non_terminal_when_no_action(self) -> None:
        """A state with action=None should not be terminal."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.5, anomaly_score=-0.1
        )
        assert not state.is_terminal()

    def test_terminal_when_action_assigned(self) -> None:
        """A state with an action should be terminal."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.5, anomaly_score=-0.1, action="block"
        )
        assert state.is_terminal()

    def test_possible_actions_returns_three(self) -> None:
        """possible_actions() should return approve, block, review."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.5, anomaly_score=-0.1
        )
        actions = state.possible_actions()
        assert set(actions) == {"approve", "block", "review"}
        assert len(actions) == 3

    def test_possible_actions_returns_copy(self) -> None:
        """possible_actions() should return a new list each time (not a reference)."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.5, anomaly_score=-0.1
        )
        actions1 = state.possible_actions()
        actions2 = state.possible_actions()
        assert actions1 is not actions2


class TestCostFunction:
    """Tests for cost computation."""

    def test_approve_cost_high_fraud(self, cost_function: CostFunction) -> None:
        """Approving a high-fraud transaction should have high expected cost.
        E[cost] = 0.9 × 1000 = 900.0"""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.9, anomaly_score=-0.5
        )
        cost = cost_function.compute(state, "approve")
        assert cost == pytest.approx(900.0)

    def test_block_cost_low_fraud(self, cost_function: CostFunction) -> None:
        """Blocking a low-fraud transaction should cost (1-p) × FP_cost.
        E[cost] = 0.99 × 50 = 49.5"""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.01, anomaly_score=0.1
        )
        cost = cost_function.compute(state, "block")
        assert cost == pytest.approx(49.5)

    def test_review_cost_is_fixed(self, cost_function: CostFunction) -> None:
        """Review cost should always be the fixed review_cost regardless of fraud probability."""
        for p in [0.0, 0.3, 0.5, 0.9, 1.0]:
            state = DecisionState(
                transaction_id=1, fraud_probability=p, anomaly_score=-0.1
            )
            cost = cost_function.compute(state, "review")
            assert cost == pytest.approx(20.0)

    def test_unknown_action_raises_error(self, cost_function: CostFunction) -> None:
        """An unrecognized action should raise ValueError."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.5, anomaly_score=-0.1
        )
        with pytest.raises(ValueError, match="Unknown action"):
            cost_function.compute(state, "escalate")

    def test_describe_returns_nonempty_string(self, cost_function: CostFunction) -> None:
        """describe() should return a non-empty string with cost configuration."""
        description = cost_function.describe()
        assert isinstance(description, str)
        assert len(description) > 0
        assert "1,000" in description  # fraud_loss formatted


class TestAStarDecisionEngine:
    """Tests for the A* search decision engine."""

    def test_high_fraud_returns_block(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """High fraud probability (0.9) should result in 'block'."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.9, anomaly_score=-0.5
        )
        result = decision_engine.search(state)
        assert result.action == "block"
        assert result.is_terminal()

    def test_low_fraud_returns_approve(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """Very low fraud probability (0.01) should result in 'approve'."""
        state = DecisionState(
            transaction_id=2, fraud_probability=0.01, anomaly_score=0.2
        )
        result = decision_engine.search(state)
        assert result.action == "approve"

    def test_medium_fraud_returns_review(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """Medium fraud probability (0.3) should result in 'review'.
        approve cost = 0.3 × 1000 = 300
        block cost = 0.7 × 50 = 35
        review cost = 20 → cheapest"""
        state = DecisionState(
            transaction_id=3, fraud_probability=0.3, anomaly_score=-0.1
        )
        result = decision_engine.search(state)
        assert result.action == "review"

    def test_result_preserves_transaction_id(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """The result state should preserve the original transaction_id."""
        state = DecisionState(
            transaction_id=42, fraud_probability=0.5, anomaly_score=-0.1
        )
        result = decision_engine.search(state)
        assert result.transaction_id == 42

    def test_result_preserves_fraud_probability(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """The result state should preserve the original fraud_probability."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.75, anomaly_score=-0.3
        )
        result = decision_engine.search(state)
        assert result.fraud_probability == 0.75

    def test_boundary_approve_to_review(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """At the boundary p=0.02, approve cost = review cost = 20.
        Just above boundary (p=0.03), review should win."""
        state = DecisionState(
            transaction_id=1, fraud_probability=0.03, anomaly_score=0.1
        )
        result = decision_engine.search(state)
        assert result.action == "review"
