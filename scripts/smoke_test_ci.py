"""CI smoke test for the FastAPI service with lifespan/startup enabled."""
from __future__ import annotations

from fastapi.testclient import TestClient

from dashboard.backend.main import app

SAMPLE = {
    "Time": 406.0,
    "Amount": 149.62,
}
SAMPLE.update({f"V{i}": 0.0 for i in range(1, 29)})


def main() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        body = health.json()
        assert body["model_loaded"] is True
        assert body["readiness"] is True

        prediction = client.post("/predict", json={"features": SAMPLE})
        assert prediction.status_code == 200, prediction.text
        result = prediction.json()
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["prediction"] in (0, 1)
        assert "threshold" in result

        explanation = client.get("/explain/0")
        assert explanation.status_code == 200, explanation.text

    print("CI smoke test passed.")


if __name__ == "__main__":
    main()
