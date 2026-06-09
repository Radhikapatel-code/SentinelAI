"""
Tests for the FastAPI backend.

Validates:
- Health check endpoint returns 200 with correct body
- Analyze endpoint processes valid transactions
- Response schema matches TransactionResponse model
- Decision values are within expected set
- Fraud probability is in [0, 1]
- Invalid input returns 422 validation error
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def valid_transaction() -> dict:
    """A valid transaction payload for POST /analyze."""
    return {
        "transaction_id": 99,
        "amount": 500.0,
        "transaction_time": 14,
        "location_change": 0,
        "device_change": 0,
        "merchant_risk": 0.05,
    }


@pytest.fixture
def fraudulent_payload() -> dict:
    """A suspicious transaction payload for POST /analyze."""
    return {
        "transaction_id": 100,
        "amount": 92000.0,
        "transaction_time": 2,
        "location_change": 1,
        "device_change": 1,
        "merchant_risk": 0.95,
    }


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_contains_status(self, client: TestClient) -> None:
        """GET /health response should include status: healthy."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_body_contains_version(self, client: TestClient) -> None:
        """GET /health response should include a version string."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)


class TestAnalyzeEndpoint:
    """Tests for POST /analyze."""

    def test_analyze_returns_200(
        self, client: TestClient, valid_transaction: dict
    ) -> None:
        """POST /analyze with a valid payload should return HTTP 200."""
        response = client.post("/analyze", json=valid_transaction)
        assert response.status_code == 200

    def test_analyze_response_has_required_fields(
        self, client: TestClient, valid_transaction: dict
    ) -> None:
        """Response must contain all TransactionResponse fields."""
        response = client.post("/analyze", json=valid_transaction)
        data = response.json()
        required_fields = {
            "transaction_id",
            "fraud_probability",
            "anomaly_score",
            "decision",
            "explanation",
            "feature_contributions",
        }
        assert required_fields.issubset(data.keys())

    def test_analyze_decision_is_valid_action(
        self, client: TestClient, valid_transaction: dict
    ) -> None:
        """decision field must be one of: approve, block, review."""
        response = client.post("/analyze", json=valid_transaction)
        data = response.json()
        assert data["decision"] in ["approve", "block", "review"]

    def test_analyze_fraud_probability_in_range(
        self, client: TestClient, valid_transaction: dict
    ) -> None:
        """fraud_probability should be between 0.0 and 1.0."""
        response = client.post("/analyze", json=valid_transaction)
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_analyze_feature_contributions_has_features(
        self, client: TestClient, valid_transaction: dict
    ) -> None:
        """feature_contributions should contain the 5 model features."""
        response = client.post("/analyze", json=valid_transaction)
        data = response.json()
        contributions = data["feature_contributions"]
        expected_features = {
            "amount",
            "transaction_time",
            "location_change",
            "device_change",
            "merchant_risk",
        }
        assert set(contributions.keys()) == expected_features

    def test_analyze_invalid_input_returns_422(self, client: TestClient) -> None:
        """POST /analyze with missing required fields should return 422."""
        response = client.post("/analyze", json={"transaction_id": 1})
        assert response.status_code == 422
