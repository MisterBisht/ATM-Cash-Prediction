
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ATM Cash Prediction API is running",
        "model": "atm-cash-prediction",
        "alias": "champion"
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_prediction(client):
    payload = {
        "nearby_shops": 15,
        "days_to_salary": 5,
        "footfall_index": 0.75,
        "distance_to_market_km": 1.5,
        "avg_ticket_rs": 250,
        "other_atms_1km": 3,
        "atm_type": "standalone",
        "area": "market"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_withdrawal_thousand_rs" in result
    assert result["unit"] == "thousand INR"

    assert isinstance(
        result["predicted_withdrawal_thousand_rs"], float
    )
