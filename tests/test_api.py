from fastapi.testclient import TestClient

from ai4i_mlops.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict(trained_model):
    data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_failure"] <= 1
    assert 0 <= result["probability_no_failure"] <= 1


def test_invalid_type(trained_model):
    data = {
        "type": "X",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422