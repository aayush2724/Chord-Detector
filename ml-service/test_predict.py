import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _model_loaded():
    resp = client.get("/health")
    return resp.json().get("model_loaded", False)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "supported_chords" in data


@pytest.mark.skipif(not _model_loaded(), reason="Model not loaded")
def test_get_classes():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) > 0


@pytest.mark.skipif(not _model_loaded(), reason="Model not loaded")
def test_predict_valid_landmarks():
    landmarks = [0.5] * 63
    response = client.post("/predict", json={"landmarks": landmarks})
    assert response.status_code == 200
    data = response.json()
    assert "chord" in data
    assert "confidence" in data
    assert "all_probs" in data
    assert 0 <= data["confidence"] <= 1


@pytest.mark.skipif(not _model_loaded(), reason="Model not loaded")
def test_classes_returns_list():
    response = client.get("/classes")
    classes = response.json()["classes"]
    assert isinstance(classes, list)


def test_predict_wrong_length():
    response = client.post("/predict", json={"landmarks": [0.5] * 10})
    assert response.status_code == 422


def test_predict_empty():
    response = client.post("/predict", json={"landmarks": []})
    assert response.status_code == 422


def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422
