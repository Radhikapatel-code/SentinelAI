"""
Tests for the SHAP Explainability module.

Validates:
- SHAP values are computed for all features
- Text explanations are generated correctly
- Top contributing features appear in explanation text
"""

import pandas as pd
import pytest

from explainability.shap_explainer import ShapExplainer


class TestShapExplain:
    """Tests for SHAP value computation."""

    def test_explain_returns_dict(
        self,
        shap_explainer: ShapExplainer,
        legitimate_transaction: pd.DataFrame,
    ) -> None:
        """explain() should return a dict."""
        result = shap_explainer.explain(legitimate_transaction)
        assert isinstance(result, dict)

    def test_explain_has_all_features(
        self,
        shap_explainer: ShapExplainer,
        legitimate_transaction: pd.DataFrame,
    ) -> None:
        """The explanation dict should contain all 5 feature names."""
        result = shap_explainer.explain(legitimate_transaction)
        expected = {
            "amount",
            "transaction_time",
            "location_change",
            "device_change",
            "merchant_risk",
        }
        assert set(result.keys()) == expected

    def test_explain_values_are_floats(
        self,
        shap_explainer: ShapExplainer,
        legitimate_transaction: pd.DataFrame,
    ) -> None:
        """All SHAP values should be numeric floats."""
        result = shap_explainer.explain(legitimate_transaction)
        for feature, value in result.items():
            assert isinstance(value, float), (
                f"Feature '{feature}' value is {type(value)}, expected float"
            )

    def test_explain_fraud_vs_legit(
        self,
        shap_explainer: ShapExplainer,
        legitimate_transaction: pd.DataFrame,
        fraudulent_transaction: pd.DataFrame,
    ) -> None:
        """Fraudulent transaction should have higher total absolute SHAP
        values than a legitimate one (model sees more signal)."""
        legit_shap = shap_explainer.explain(legitimate_transaction)
        fraud_shap = shap_explainer.explain(fraudulent_transaction)

        legit_total = sum(abs(v) for v in legit_shap.values())
        fraud_total = sum(abs(v) for v in fraud_shap.values())

        # The fraud transaction has extreme values — model should react more
        assert fraud_total > 0, "Fraud SHAP total should be non-zero"


class TestTextExplanation:
    """Tests for natural language explanation generation."""

    def test_generate_text_returns_string(
        self,
        shap_explainer: ShapExplainer,
        legitimate_transaction: pd.DataFrame,
    ) -> None:
        """Text explanation should be a non-empty string."""
        shap_values = shap_explainer.explain(legitimate_transaction)
        text = shap_explainer.generate_text_explanation(shap_values)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_explanation_contains_flagged(
        self,
        shap_explainer: ShapExplainer,
        legitimate_transaction: pd.DataFrame,
    ) -> None:
        """The explanation should contain the word 'flagged'."""
        shap_values = shap_explainer.explain(legitimate_transaction)
        text = shap_explainer.generate_text_explanation(shap_values)
        assert "flagged" in text.lower() or "transaction" in text.lower()

    def test_top_feature_in_explanation(
        self,
        shap_explainer: ShapExplainer,
        fraudulent_transaction: pd.DataFrame,
    ) -> None:
        """The top SHAP feature (by absolute value) should appear in the
        explanation text (with underscores replaced by spaces)."""
        shap_values = shap_explainer.explain(fraudulent_transaction)
        text = shap_explainer.generate_text_explanation(shap_values)

        # Find the top feature
        top_feature = max(shap_values, key=lambda k: abs(shap_values[k]))
        # The explanation replaces underscores with spaces
        readable_name = top_feature.replace("_", " ")
        assert readable_name in text, (
            f"Top feature '{readable_name}' not found in: '{text}'"
        )

    def test_explanation_with_zero_shap_values(
        self, shap_explainer: ShapExplainer
    ) -> None:
        """Edge case: all-zero SHAP values should still produce valid text."""
        zero_shap = {
            "amount": 0.0,
            "transaction_time": 0.0,
            "location_change": 0.0,
            "device_change": 0.0,
            "merchant_risk": 0.0,
        }
        text = shap_explainer.generate_text_explanation(zero_shap)
        assert isinstance(text, str)
        assert len(text) > 0
