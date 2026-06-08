"""
SentinelAI SHAP Explainability Module.

Generates feature attribution explanations using SHAP (SHapley Additive
exPlanations). Provides both structured SHAP values and natural language
explanations constructed programmatically from the top contributing features.

Example:
    >>> explainer = ShapExplainer(trained_pipeline)
    >>> shap_vals = explainer.explain(transaction_df)
    >>> text = explainer.generate_text_explanation(shap_vals, transaction_df)
    >>> print(text)
    "Transaction flagged: unusually high amount (₹92,000) increased risk,
     combined with device change and rapid location change."
"""

from typing import Optional

import shap
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# Feature Description Mapping
# ──────────────────────────────────────────────
_FEATURE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "amount": {
        "high": "unusually high transaction amount",
        "low": "very low transaction amount",
    },
    "transaction_time": {
        "high": "transaction at unusual late-night hour",
        "low": "transaction during off-peak hours",
    },
    "location_change": {
        "high": "rapid location change detected",
        "low": "consistent transaction location",
    },
    "device_change": {
        "high": "new or unfamiliar device used",
        "low": "transaction from known device",
    },
    "merchant_risk": {
        "high": "high-risk merchant category",
        "low": "trusted merchant",
    },
}


class ShapExplainer:
    """Generates explainability insights using SHAP with a callable model wrapper.

    This class wraps a trained sklearn pipeline and uses SHAP's Explainer
    to compute feature attributions. It also provides methods to convert
    raw SHAP values into human-readable natural language explanations.

    Attributes:
        pipeline: The trained sklearn pipeline (scaler + classifier).
        feature_names: List of feature column names.
        explainer: The SHAP Explainer instance.
    """

    def __init__(self, trained_pipeline: object) -> None:
        """Initialize the SHAP explainer with a trained model pipeline.

        Args:
            trained_pipeline: A fitted sklearn Pipeline with a
                predict_proba method.
        """
        self.pipeline = trained_pipeline

        self.feature_names: list[str] = [
            "amount",
            "transaction_time",
            "location_change",
            "device_change",
            "merchant_risk",
        ]

        # Background data for SHAP (small synthetic baseline)
        self.background: pd.DataFrame = pd.DataFrame(
            np.zeros((1, len(self.feature_names))),
            columns=self.feature_names,
        )

        # Wrap pipeline into a callable function
        def model_predict(X: np.ndarray) -> np.ndarray:
            """Predict fraud probability for SHAP."""
            return self.pipeline.predict_proba(X)[:, 1]

        self.explainer: shap.Explainer = shap.Explainer(
            model_predict,
            self.background,
            feature_names=self.feature_names,
        )

    def explain(self, transaction: pd.DataFrame) -> dict[str, float]:
        """Generate SHAP explanation for a single transaction.

        Computes the SHAP values indicating each feature's contribution
        to the model's fraud probability prediction.

        Args:
            transaction: A single-row DataFrame with the transaction features.

        Returns:
            dict[str, float]: Mapping of feature name to its SHAP value.
                Positive values increase fraud probability;
                negative values decrease it.
        """
        shap_values = self.explainer(transaction[self.feature_names])

        values: np.ndarray = shap_values.values[0]

        explanation: dict[str, float] = {
            feature: float(value)
            for feature, value in zip(self.feature_names, values)
        }

        return explanation

    def generate_text_explanation(
        self,
        shap_values: dict[str, float],
        transaction: Optional[pd.DataFrame] = None,
    ) -> str:
        """Convert SHAP values into a descriptive natural language explanation.

        Constructs a human-readable sentence from the top 3 most impactful
        features, using directional descriptions (e.g., "unusually high
        transaction amount increased risk").

        Args:
            shap_values: Feature name → SHAP value mapping from explain().
            transaction: Optional transaction DataFrame for including
                actual values in the explanation (e.g., "amount of ₹92,000").

        Returns:
            str: A natural language explanation of why the transaction
                was flagged or approved.
        """
        # Sort features by absolute SHAP impact (descending)
        sorted_features: list[tuple[str, float]] = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_features: list[tuple[str, float]] = sorted_features[:3]

        reasons: list[str] = []
        for feature, shap_value in top_features:
            direction = "high" if shap_value > 0 else "low"

            # Get descriptive text for this feature
            desc_map = _FEATURE_DESCRIPTIONS.get(feature, {})
            description = desc_map.get(
                direction,
                f"{feature.replace('_', ' ')} {'increased' if shap_value > 0 else 'decreased'} risk",
            )

            # Append actual value if transaction data is available
            if transaction is not None and feature in transaction.columns:
                actual_value = transaction[feature].values[0]
                if feature == "amount":
                    description += f" (₹{actual_value:,.0f})"
                elif feature in ("location_change", "device_change"):
                    if actual_value == 1:
                        description += " (flagged)"
                elif feature == "merchant_risk":
                    description += f" (score: {actual_value:.2f})"

            # Add directionality
            if shap_value > 0:
                reasons.append(f"{description} increased risk")
            else:
                reasons.append(f"{description} decreased risk")

        # Build the final sentence
        if not reasons:
            return "Transaction analyzed — no significant risk factors identified."

        explanation = "Transaction flagged primarily due to: " + ", ".join(reasons) + "."
        return explanation
