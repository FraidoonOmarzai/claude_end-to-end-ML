"""
API Tests
=========

Tests for the FastAPI endpoints using pytest and httpx.

STUDY NOTE:
-----------
Testing ML APIs is crucial because:
- Validates the integration works
- Catches bugs before deployment
- Documents expected behavior
- Enables confident refactoring

Run tests:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.main import app


# =============================================================================
# TEST CLIENT SETUP
# =============================================================================
"""
STUDY NOTE: TestClient
----------------------
FastAPI's TestClient (based on httpx) allows testing
without running an actual server.

Benefits:
- Fast tests
- No network required
- Easy to mock dependencies
"""

client = TestClient(app)


# =============================================================================
# TESTS
# =============================================================================

class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_welcome_message(self):
        """Test that root endpoint returns welcome message."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Welcome" in data["message"]
        assert "version" in data


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_status(self):
        """Test health endpoint returns status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""

    def test_model_info_returns_metadata(self):
        """Test model info endpoint returns model metadata."""
        response = client.get("/model/info")

        # May fail if model not trained yet
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "version" in data
            assert "feature_names" in data
            assert "target_names" in data


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_valid_input_setosa(self):
        """Test prediction with valid input (expected: setosa)."""
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert data["predicted_class"] == "setosa"
            assert data["confidence"] > 0.5

    def test_predict_valid_input_virginica(self):
        """Test prediction with valid input (expected: virginica)."""
        response = client.post(
            "/predict",
            json={
                "sepal_length": 6.7,
                "sepal_width": 3.0,
                "petal_length": 5.2,
                "petal_width": 2.3
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert data["predicted_class"] == "virginica"

    def test_predict_missing_field_returns_422(self):
        """Test that missing fields return validation error."""
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5
                # Missing petal_length and petal_width
            }
        )

        assert response.status_code == 422  # Validation error

    def test_predict_invalid_type_returns_422(self):
        """Test that invalid types return validation error."""
        response = client.post(
            "/predict",
            json={
                "sepal_length": "not a number",  # Should be float
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )

        assert response.status_code == 422

    def test_predict_negative_value_returns_422(self):
        """Test that negative values return validation error."""
        response = client.post(
            "/predict",
            json={
                "sepal_length": -1.0,  # Negative not allowed
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        )

        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the batch prediction endpoint."""

    def test_batch_predict_multiple_samples(self):
        """Test batch prediction with multiple samples."""
        response = client.post(
            "/predict/batch",
            json={
                "samples": [
                    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                    {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
                    {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5}
                ]
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "count" in data
            assert data["count"] == 3
            assert len(data["predictions"]) == 3

    def test_batch_predict_empty_returns_422(self):
        """Test that empty batch returns validation error."""
        response = client.post(
            "/predict/batch",
            json={"samples": []}
        )

        assert response.status_code == 422


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
