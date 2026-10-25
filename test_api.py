from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/health"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post("/predict", json={
        "study_hours": 4,
        "attendance": 85,
        "previous_score": 70,
        "assignment_score": 80
    })
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["predicted_score"] <= 100
    assert body["predicted_result"] in ["Pass", "Fail"]
