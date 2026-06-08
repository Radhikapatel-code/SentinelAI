"""
Tests for the A* Decision Engine, CostFunction, and DecisionState.

Validates:
- Cost function computes correct expected costs for each action
- A* search returns correct decisions for high/low/borderline risk
- DecisionState correctly identifies terminal states
"""

import pytest

from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine


# ──────────────────────────────────────────────
# DecisionState Tests
# ──────────────────────────────────────────────


class TestDecisionState:
    """Tests for the DecisionState dataclass."""

    def test_initial_state_is_not_terminal(self) -> None:
        """A state with no action should not be terminal."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        assert not state.is_terminal()

    def test_state_with_action_is_terminal(self) -> None:
        """A state with an action assigned should be terminal."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
            action="block",
        )
        assert state.is_terminal()

    def test_possible_actions(self) -> None:
        """Should return exactly three actions: approve, block, review."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        actions = state.possible_actions()
        assert set(actions) == {"approve", "block", "review"}

    def test_state_is_frozen(self) -> None:
        """DecisionState is a frozen dataclass — mutation should raise."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        with pytest.raises(AttributeError):
            state.action = "block"  # type: ignore[misc]


# ──────────────────────────────────────────────
# CostFunction Tests
# ──────────────────────────────────────────────


class TestCostFunction:
    """Tests for expected cost computation."""

    def test_approve_cost_formula(self, cost_function: CostFunction) -> None:
        """Approve cost = fraud_probability × fraud_loss.

        For p_fraud=0.8, fraud_loss=1000 → cost = 800.
        """
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.8,
            anomaly_score=-0.2,
        )
        cost = cost_function.compute(state, "approve")
        assert cost == pytest.approx(0.8 * 1000, rel=1e-6)

    def test_block_cost_formula(self, cost_function: CostFunction) -> None:
        """Block cost = (1 - fraud_probability) × false_positive_cost.

        For p_fraud=0.8, false_positive_cost=50 → cost = 10.
        """
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.8,
            anomaly_score=-0.2,
        )
        cost = cost_function.compute(state, "block")
        assert cost == pytest.approx(0.2 * 50, rel=1e-6)

    def test_review_cost_is_fixed(self, cost_function: CostFunction) -> None:
        """Review cost is always the fixed review_cost (20), regardless of
        fraud probability."""
        for p_fraud in [0.0, 0.5, 1.0]:
            state = DecisionState(
                transaction_id=1,
                fraud_probability=p_fraud,
                anomaly_score=-0.1,
            )
            cost = cost_function.compute(state, "review")
            assert cost == pytest.approx(20.0)

    def test_unknown_action_raises_value_error(
        self, cost_function: CostFunction
    ) -> None:
        """An unrecognized action should raise ValueError."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        with pytest.raises(ValueError, match="Unknown action"):
            cost_function.compute(state, "escalate")

    def test_zero_fraud_probability_approve_is_free(
        self, cost_function: CostFunction
    ) -> None:
        """If fraud probability is 0, approving has zero cost."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.0,
            anomaly_score=0.1,
        )
        cost = cost_function.compute(state, "approve")
        assert cost == pytest.approx(0.0)

    def test_certain_fraud_block_is_free(
        self, cost_function: CostFunction
    ) -> None:
        """If fraud probability is 1.0, blocking has zero false-positive cost."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=1.0,
            anomaly_score=-0.5,
        )
        cost = cost_function.compute(state, "block")
        assert cost == pytest.approx(0.0)


# ──────────────────────────────────────────────
# AStarDecisionEngine Tests
# ──────────────────────────────────────────────


class TestAStarDecisionEngine:
    """Tests for the A* search decision logic."""

    def test_high_risk_transaction_blocked(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """A transaction with very high fraud probability should be blocked.

        At p_fraud=0.95:
        - approve cost = 0.95 × 1000 = 950
        - block cost   = 0.05 × 50  = 2.5
        - review cost  = 20
        Minimum is block (2.5).
        """
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.95,
            anomaly_score=-0.3,
        )
        result = decision_engine.search(state)
        assert result.action == "block"

    def test_low_risk_transaction_approved(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """A transaction with very low fraud probability should be approved.

        At p_fraud=0.01:
        - approve cost = 0.01 × 1000 = 10
        - block cost   = 0.99 × 50  = 49.5
        - review cost  = 20
        Minimum is approve (10).
        """
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.01,
            anomaly_score=0.1,
        )
        result = decision_engine.search(state)
        assert result.action == "approve"

    def test_borderline_transaction_reviewed(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """A borderline transaction should be sent to review.

        With default costs (fraud_loss=1000, false_positive_cost=50, review_cost=20):
        - review is chosen when: review_cost < both approve_cost and block_cost
        - approve_cost = p × 1000 > 20 when p > 0.02
        - block_cost = (1-p) × 50 > 20 when p < 0.6

        So for p_fraud in (0.02, 0.6), review should be cheapest.
        Let's use p_fraud=0.3:
        - approve cost = 0.3 × 1000 = 300
        - block cost   = 0.7 × 50  = 35
        - review cost  = 20
        Minimum is review (20).
        """
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.3,
            anomaly_score=-0.05,
        )
        result = decision_engine.search(state)
        assert result.action == "review"

    def test_result_is_terminal(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """The returned state should always be terminal (action assigned)."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        result = decision_engine.search(state)
        assert result.is_terminal()

    def test_preserves_transaction_id(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """The result should carry the same transaction_id as the input."""
        state = DecisionState(
            transaction_id=42,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        result = decision_engine.search(state)
        assert result.transaction_id == 42

    def test_decision_is_valid_action(
        self, decision_engine: AStarDecisionEngine
    ) -> None:
        """The chosen action must be one of the valid actions."""
        state = DecisionState(
            transaction_id=1,
            fraud_probability=0.5,
            anomaly_score=-0.1,
        )
        result = decision_engine.search(state)
        assert result.action in {"approve", "block", "review"}
