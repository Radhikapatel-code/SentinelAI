"""
SentinelAI Decision Engine Package.

Provides the A* search-based decision engine for fraud handling:
- DecisionState: Immutable state representation for transactions
- CostFunction: Expected cost computation for each action
- AStarDecisionEngine: Optimal action selection via A* search
"""

from decision_engine.state import DecisionState
from decision_engine.cost_function import CostFunction
from decision_engine.a_star import AStarDecisionEngine

__all__ = ["DecisionState", "CostFunction", "AStarDecisionEngine"]
