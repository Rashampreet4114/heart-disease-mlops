from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Disease", "No Disease")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_invalid_input():
    bad_patient = dict(SAMPLE_PATIENT)
    bad_patient["sex"] = 5  # out of allowed range [0, 1]
    response = client.post("/predict", json=bad_patient)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete = dict(SAMPLE_PATIENT)
    del incomplete["age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_metrics_endpoint_tracks_requests():
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["requests_total"] > 0
