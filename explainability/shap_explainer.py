import shap
import pandas as pd
import numpy as np


class ShapExplainer:
    """
    Generates explainability insights using SHAP with a callable model wrapper.
    """

    def __init__(self, trained_pipeline):
        """
        trained_pipeline: sklearn pipeline with scaler + classifier
        """
        self.pipeline = trained_pipeline

        self.feature_names = [
            "amount",
            "transaction_time",
            "location_change",
            "device_change",
            "merchant_risk",
        ]

        # Background data for SHAP (small synthetic baseline)
        self.background = pd.DataFrame(
            np.zeros((1, len(self.feature_names))),
            columns=self.feature_names,
        )

        # Wrap pipeline into a callable function
        def model_predict(X):
            return self.pipeline.predict_proba(X)[:, 1]

        self.explainer = shap.Explainer(
            model_predict,
            self.background,
            feature_names=self.feature_names,
        )

    def explain(self, transaction: pd.DataFrame) -> dict:
        """
        Generate SHAP explanation for a single transaction.
        """
        shap_values = self.explainer(transaction[self.feature_names])

        values = shap_values.values[0]

        explanation = {
            feature: float(value)
            for feature, value in zip(self.feature_names, values)
        }

        return explanation

    def generate_text_explanation(self, shap_values: dict) -> str:
        """
        Convert SHAP values into a natural language explanation.
        """
        sorted_features = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_features = sorted_features[:3]

        reasons = [
            f"{feature.replace('_', ' ')} impact"
            for feature, _ in top_features
        ]

        return (
            "Transaction flagged primarily due to: "
            + ", ".join(reasons)
            + "."
        )
