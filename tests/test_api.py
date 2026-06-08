"""
Tests for the FastAPI /analyze endpoint.

Uses FastAPI's TestClient to validate:
- Correct HTTP status codes
- Response schema completeness
- Decision value constraints
- Input validation (422 on bad input)
"""

import sys
import os
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI TestClient.

    Returns:
        TestClient: A test client bound to the SentinelAI app.
    """
    return TestClient(app)


@pytest.fixture
def valid_payload() -> dict:
    """A valid transaction payload for the /analyze endpoint.

    Returns:
        dict: Transaction data with all required fields.
    """
    return {
        "transaction_id": 1,
        "amount": 5000.0,
        "transaction_time": 14,
        "location_change": 0,
        "device_change": 0,
        "merchant_risk": 0.2,
    }


@pytest.fixture
def high_risk_payload() -> dict:
    """A high-risk transaction payload.

    Returns:
        dict: Transaction data with all fraud indicators.
    """
    return {
        "transaction_id": 2,
        "amount": 89000.0,
        "transaction_time": 2,
        "location_change": 1,
        "device_change": 1,
        "merchant_risk": 0.95,
    }


class TestAnalyzeEndpoint:
    """Tests for POST /analyze."""

    def test_returns_200_for_valid_input(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """A valid transaction should return HTTP 200."""
        response = client.post("/analyze", json=valid_payload)
        assert response.status_code == 200

    def test_response_contains_all_keys(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Response JSON must contain all expected fields."""
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        expected_keys = {
            "transaction_id",
            "fraud_probability",
            "anomaly_score",
            "decision",
            "explanation",
            "feature_contributions",
        }
        assert expected_keys.issubset(data.keys()), (
            f"Missing keys: {expected_keys - data.keys()}"
        )

    def test_decision_is_valid_value(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Decision must be one of: approve, block, review."""
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        assert data["decision"] in {"approve", "block", "review"}

    def test_fraud_probability_is_bounded(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Fraud probability should be between 0 and 1."""
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_explanation_is_nonempty_string(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Explanation should be a non-empty string."""
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        assert isinstance(data["explanation"], str)
        assert len(data["explanation"]) > 0

    def test_feature_contributions_has_all_features(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """Feature contributions should include all 5 input features."""
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        expected_features = {
            "amount",
            "transaction_time",
            "location_change",
            "device_change",
            "merchant_risk",
        }
        assert expected_features == set(data["feature_contributions"].keys())

    def test_high_risk_transaction_response(
        self, client: TestClient, high_risk_payload: dict
    ) -> None:
        """A high-risk transaction should have high fraud probability."""
        response = client.post("/analyze", json=high_risk_payload)
        data = response.json()
        # With a clearly fraudulent transaction, probability should be elevated
        assert data["fraud_probability"] > 0.3

    def test_transaction_id_echoed_back(
        self, client: TestClient, valid_payload: dict
    ) -> None:
        """The response should echo the input transaction_id."""
        response = client.post("/analyze", json=valid_payload)
        data = response.json()
        assert data["transaction_id"] == valid_payload["transaction_id"]


class TestAnalyzeEndpointValidation:
    """Tests for input validation (422 errors)."""

    def test_missing_amount_returns_422(
        self, client: TestClient
    ) -> None:
        """Omitting a required field should return 422 Unprocessable Entity."""
        payload = {
            "transaction_id": 1,
            # "amount" is missing
            "transaction_time": 14,
            "location_change": 0,
            "device_change": 0,
            "merchant_risk": 0.2,
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(
        self, client: TestClient
    ) -> None:
        """An empty request body should return 422."""
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_wrong_type_returns_422(
        self, client: TestClient
    ) -> None:
        """Passing a string where a number is expected should return 422."""
        payload = {
            "transaction_id": "not_a_number",
            "amount": 5000.0,
            "transaction_time": 14,
            "location_change": 0,
            "device_change": 0,
            "merchant_risk": 0.2,
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 422
