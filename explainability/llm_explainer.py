"""
SentinelAI LLM Explanation Layer.

Provides optional LLM-powered natural language explanations for fraud
decisions. Uses OpenAI's API to generate contextual, business-friendly
explanations from SHAP values and transaction data.

Falls back gracefully to SHAP-based text explanations when:
- No API key is configured
- The LLM API call fails
- LLM is disabled in config

Usage:
    from explainability.llm_explainer import LLMExplainer

    explainer = LLMExplainer()
    explanation = explainer.explain(
        shap_values={"amount": 0.35, "device_change": 0.22, ...},
        transaction={"amount": 89000, "device_change": 1, ...},
        decision="block",
    )
"""

import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an AI fraud analyst for SentinelAI, a financial fraud detection system.
Your job is to explain fraud detection decisions in clear, professional language
that a compliance officer or customer service agent can understand.

Keep explanations concise (2-3 sentences max). Be specific about which factors
contributed to the decision and why. Use professional financial language."""

_USER_PROMPT_TEMPLATE = """Explain this fraud detection decision:

Transaction Details:
{transaction_details}

SHAP Feature Contributions (positive = increases fraud risk):
{shap_details}

Decision: {decision}

Provide a concise, professional explanation of why this decision was made."""


class LLMExplainer:
    """Generates LLM-powered natural language explanations for fraud decisions.

    This class integrates with OpenAI's API to produce contextual,
    business-friendly explanations. It includes automatic fallback
    to SHAP-based text when the LLM is unavailable.

    Attributes:
        api_key: The OpenAI API key (from env or config).
        model: The LLM model identifier.
        max_tokens: Maximum response length.
        enabled: Whether LLM explanations are active.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 150,
        enabled: bool = True,
    ) -> None:
        """Initialize the LLM explainer.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY
                environment variable.
            model: The LLM model to use for explanations.
            max_tokens: Maximum number of tokens in the response.
            enabled: Set to False to skip LLM calls entirely.
        """
        self.api_key: Optional[str] = api_key or os.environ.get("OPENAI_API_KEY")
        self.model: str = model
        self.max_tokens: int = max_tokens
        self.enabled: bool = enabled and (self.api_key is not None)

        if not self.enabled:
            logger.info(
                "LLM explainer disabled (no API key or explicitly disabled). "
                "Will fall back to SHAP-based explanations."
            )

    def is_available(self) -> bool:
        """Check if the LLM explainer is configured and ready.

        Returns:
            bool: True if an API key is set and LLM is enabled.
        """
        return self.enabled and self.api_key is not None

    def explain(
        self,
        shap_values: dict[str, float],
        transaction: dict[str, Any],
        decision: str,
        fallback_text: Optional[str] = None,
    ) -> str:
        """Generate an LLM-powered explanation for a fraud decision.

        If the LLM is unavailable or the API call fails, returns the
        fallback_text instead.

        Args:
            shap_values: Feature name → SHAP value mapping.
            transaction: Transaction data as a dictionary.
            decision: The decision made ("approve", "block", or "review").
            fallback_text: Text to return if LLM is unavailable.
                If None, generates a basic explanation from SHAP values.

        Returns:
            str: A natural language explanation of the decision.
        """
        if not self.is_available():
            return fallback_text or self._basic_fallback(shap_values, decision)

        try:
            return self._call_llm(shap_values, transaction, decision)
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}. Using fallback.")
            return fallback_text or self._basic_fallback(shap_values, decision)

    def _call_llm(
        self,
        shap_values: dict[str, float],
        transaction: dict[str, Any],
        decision: str,
    ) -> str:
        """Make the actual API call to OpenAI.

        Args:
            shap_values: Feature contributions.
            transaction: Transaction details.
            decision: The fraud decision.

        Returns:
            str: The LLM-generated explanation.

        Raises:
            ImportError: If the openai package is not installed.
            Exception: If the API call fails.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for LLM explanations. "
                "Install it with: pip install openai"
            )

        client = OpenAI(api_key=self.api_key)

        # Format transaction details
        transaction_details = "\n".join(
            f"  - {k}: {v}" for k, v in transaction.items()
        )

        # Format SHAP values
        sorted_shap = sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )
        shap_details = "\n".join(
            f"  - {feature}: {value:+.4f}"
            for feature, value in sorted_shap
        )

        # Build the prompt
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            transaction_details=transaction_details,
            shap_details=shap_details,
            decision=decision.upper(),
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    @staticmethod
    def _basic_fallback(
        shap_values: dict[str, float],
        decision: str,
    ) -> str:
        """Generate a basic explanation without an LLM.

        Args:
            shap_values: Feature contributions.
            decision: The fraud decision.

        Returns:
            str: A simple explanation based on top SHAP features.
        """
        sorted_features = sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_3 = sorted_features[:3]

        reasons = []
        for feature, value in top_3:
            direction = "increased" if value > 0 else "decreased"
            reasons.append(f"{feature.replace('_', ' ')} {direction} risk")

        action_text = {
            "approve": "approved",
            "block": "blocked",
            "review": "flagged for manual review",
        }

        return (
            f"Transaction {action_text.get(decision, decision)} — "
            f"key factors: {', '.join(reasons)}."
        )
