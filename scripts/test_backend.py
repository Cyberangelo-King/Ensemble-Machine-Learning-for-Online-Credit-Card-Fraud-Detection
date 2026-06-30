"""
scripts/test_backend.py
pytest test suite for the Fraud Detection FastAPI backend.

Usage
-----
Without a live server (default — uses httpx.AsyncClient with ASGI transport):
    pytest scripts/test_backend.py -v

Against a live server (set BASE_URL env var):
    BASE_URL=http://localhost:8000 pytest scripts/test_backend.py -v

WebSocket tests require the optional `websocket-client` package:
    pip install websocket-client
"""

from __future__ import annotations

import os
import importlib
import pytest

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import websocket  # websocket-client package
    HAS_WEBSOCKET_CLIENT = True
except ImportError:
    HAS_WEBSOCKET_CLIENT = False

# ---------------------------------------------------------------------------
# Decide whether to run against a live server or use ASGI transport
# ---------------------------------------------------------------------------
LIVE_BASE_URL: str | None = os.environ.get("BASE_URL")  # e.g. "http://localhost:8000"
USE_LIVE_SERVER: bool = LIVE_BASE_URL is not None

# ---------------------------------------------------------------------------
# Sample payload — 30 features (V1-V28, Amount, Time)
# ---------------------------------------------------------------------------
SAMPLE_FEATURES: dict[str, float] = {
    "Time": 406.0,
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    "V5": -0.338320769942518,
    "V6": 0.462387777762292,
    "V7": 0.239598554061257,
    "V8": 0.0986979012610507,
    "V9": 0.363786969611213,
    "V10": 0.0907941719789316,
    "V11": -0.551599533260813,
    "V12": -0.617800855762348,
    "V13": -0.991389847235408,
    "V14": -0.311169353699879,
    "V15": 1.46817697209427,
    "V16": -0.470400525259478,
    "V17": 0.207971241929242,
    "V18": 0.0257905801869498,
    "V19": 0.403992960255733,
    "V20": 0.251412098239705,
    "V21": -0.018306777944153,
    "V22": 0.277837575558899,
    "V23": -0.110473910188767,
    "V24": 0.0669280749146731,
    "V25": 0.128539358273528,
    "V26": -0.189114843888824,
    "V27": 0.133558376740387,
    "V28": -0.0210530534538215,
    "Amount": 149.62,
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def sync_client():
    """
    Return a synchronous test client.

    - If BASE_URL env var is set: use httpx against a live server.
    - Otherwise: mount the ASGI app directly (no live server needed).
    """
    import httpx

    if USE_LIVE_SERVER:
        client = httpx.Client(base_url=LIVE_BASE_URL, timeout=10.0)
        yield client
        client.close()
    else:
        # Import the FastAPI app.  If artefacts are missing the startup lifespan
        # will log warnings and set state to None — endpoints that require the
        # model will return 503, which we handle gracefully in the tests below.
        try:
            from dashboard.backend.main import app
            transport = httpx.ASGITransport(app=app)
            client = httpx.Client(
                transport=transport,
                base_url="http://testserver",
                timeout=10.0,
            )
            yield client
            client.close()
        except ImportError as exc:
            pytest.skip(f"Could not import backend app: {exc}")


# =============================================================================
# Health endpoint
# =============================================================================

class TestHealth:
    def test_health_returns_200(self, sync_client):
        """GET /health must return HTTP 200 regardless of model state."""
        response = sync_client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200 from /health, got {response.status_code}: {response.text}"
        )

    def test_health_json_shape(self, sync_client):
        """Response body must have 'status' and 'model_loaded' keys."""
        response = sync_client.get("/health")
        body = response.json()
        assert "status" in body, "Missing 'status' key in /health response"
        assert "model_loaded" in body, "Missing 'model_loaded' key in /health response"

    def test_health_status_value(self, sync_client):
        """'status' field must equal 'ok'."""
        body = sync_client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_model_loaded_is_bool(self, sync_client):
        """'model_loaded' must be a boolean."""
        body = sync_client.get("/health").json()
        assert isinstance(body["model_loaded"], bool)


# =============================================================================
# Predict endpoint
# =============================================================================

class TestPredict:
    def test_predict_accepts_valid_payload(self, sync_client):
        """POST /predict with a 30-feature payload must return 200 or 503."""
        payload = {"features": SAMPLE_FEATURES}
        response = sync_client.post("/predict", json=payload)
        # 200 = model loaded and prediction made
        # 503 = model not yet loaded (graceful degradation — acceptable in CI)
        assert response.status_code in (200, 503), (
            f"Expected 200 or 503 from /predict, got {response.status_code}: {response.text}"
        )

    @pytest.mark.skipif(
        not USE_LIVE_SERVER,
        reason="Model artefact may not be present in unit-test mode; "
               "run against a live server to assert 200.",
    )
    def test_predict_returns_200_live(self, sync_client):
        """Against a live server the model must be loaded and return 200."""
        payload = {"features": SAMPLE_FEATURES}
        response = sync_client.post("/predict", json=payload)
        assert response.status_code == 200, response.text

    def test_predict_response_shape_when_200(self, sync_client):
        """If 200, response must have fraud_probability, prediction, model_loaded."""
        payload = {"features": SAMPLE_FEATURES}
        response = sync_client.post("/predict", json=payload)
        if response.status_code != 200:
            pytest.skip("Model not loaded — skipping shape assertions.")
        body = response.json()
        assert "fraud_probability" in body
        assert "prediction" in body
        assert "model_loaded" in body

    def test_predict_fraud_probability_range(self, sync_client):
        """fraud_probability must be in [0, 1]."""
        payload = {"features": SAMPLE_FEATURES}
        response = sync_client.post("/predict", json=payload)
        if response.status_code != 200:
            pytest.skip("Model not loaded — skipping range assertions.")
        body = response.json()
        prob = body["fraud_probability"]
        assert 0.0 <= prob <= 1.0, f"fraud_probability={prob} out of [0, 1]"

    def test_predict_prediction_is_binary(self, sync_client):
        """prediction must be 0 or 1."""
        payload = {"features": SAMPLE_FEATURES}
        response = sync_client.post("/predict", json=payload)
        if response.status_code != 200:
            pytest.skip("Model not loaded — skipping binary assertions.")
        body = response.json()
        assert body["prediction"] in (0, 1), f"prediction={body['prediction']} not in {{0, 1}}"

    def test_predict_rejects_empty_features(self, sync_client):
        """POST /predict with no features should return 422 (validation error)."""
        response = sync_client.post("/predict", json={})
        # FastAPI returns 422 for missing required fields
        assert response.status_code == 422, (
            f"Expected 422 for empty payload, got {response.status_code}"
        )

    def test_predict_rejects_non_numeric_features(self, sync_client):
        """POST /predict with non-numeric feature values should return 422."""
        payload = {"features": {"V1": "not_a_number", "Amount": "abc"}}
        response = sync_client.post("/predict", json=payload)
        assert response.status_code == 422


# =============================================================================
# Explain endpoint
# =============================================================================

class TestExplain:
    def test_explain_index_0(self, sync_client):
        """GET /explain/0 must return 200, 404, or 503 — never 500."""
        response = sync_client.get("/explain/0")
        assert response.status_code in (200, 404, 503), (
            f"Unexpected status from /explain/0: {response.status_code}: {response.text}"
        )

    def test_explain_response_shape_when_200(self, sync_client):
        """If 200, response must contain expected keys."""
        response = sync_client.get("/explain/0")
        if response.status_code != 200:
            pytest.skip("Explain endpoint not available (no model/data).")
        body = response.json()
        for key in ("index", "features", "shap_values", "expected_value"):
            assert key in body, f"Missing key '{key}' in /explain/0 response"

    def test_explain_shap_values_length(self, sync_client):
        """shap_values list must have same length as features list."""
        response = sync_client.get("/explain/0")
        if response.status_code != 200:
            pytest.skip("Explain endpoint not available.")
        body = response.json()
        assert len(body["shap_values"]) == len(body["features"]), (
            "shap_values and features lists have different lengths"
        )

    def test_explain_expected_value_is_numeric(self, sync_client):
        """expected_value must be a float."""
        response = sync_client.get("/explain/0")
        if response.status_code != 200:
            pytest.skip("Explain endpoint not available.")
        body = response.json()
        assert isinstance(body["expected_value"], (int, float)), (
            f"expected_value is not numeric: {body['expected_value']!r}"
        )

    def test_explain_negative_index_returns_422_or_404(self, sync_client):
        """Negative index should be rejected."""
        response = sync_client.get("/explain/-1")
        # FastAPI path param validation may return 422; our logic returns 404
        assert response.status_code in (404, 422, 503), (
            f"Expected 404/422/503 for /explain/-1, got {response.status_code}"
        )


# =============================================================================
# WebSocket stream (optional — requires websocket-client)
# =============================================================================

@pytest.mark.skipif(
    not HAS_WEBSOCKET_CLIENT,
    reason="websocket-client not installed. Install with: pip install websocket-client",
)
@pytest.mark.skipif(
    not USE_LIVE_SERVER,
    reason="WebSocket test requires a live server. Set BASE_URL env var.",
)
class TestWebSocket:
    def test_ws_stream_connects_and_receives_message(self):
        """Connect to /ws/stream, receive at least one JSON message, then close."""
        import json as _json
        import time

        ws_url = (LIVE_BASE_URL or "http://localhost:8000").replace(
            "https://", "wss://"
        ).replace("http://", "ws://") + "/ws/stream"

        ws = websocket.create_connection(ws_url, timeout=10)
        try:
            raw = ws.recv()
            msg = _json.loads(raw)
            # The message must be a dict — either a payload or an error notice
            assert isinstance(msg, dict), f"Expected JSON object, got: {raw!r}"
        finally:
            ws.close()

    def test_ws_stream_message_has_prediction_key(self):
        """When the model is loaded, messages should include 'fraud_probability'."""
        import json as _json

        ws_url = (LIVE_BASE_URL or "http://localhost:8000").replace(
            "https://", "wss://"
        ).replace("http://", "ws://") + "/ws/stream"

        ws = websocket.create_connection(ws_url, timeout=10)
        try:
            raw = ws.recv()
            msg = _json.loads(raw)
            if "error" in msg:
                pytest.skip(f"Server returned error (model not loaded?): {msg['error']}")
            assert "fraud_probability" in msg or "prediction" in msg, (
                f"Expected 'fraud_probability' or 'prediction' in WS message: {msg}"
            )
        finally:
            ws.close()


# =============================================================================
# Inline smoke-test script (also importable from CI)
# =============================================================================

def run_smoke_tests_inline() -> None:
    """
    Minimal non-pytest smoke test — useful for CI scripts that don't want pytest.
    Raises AssertionError on failure.
    """
    import httpx

    base = LIVE_BASE_URL or "http://localhost:8000"
    client = httpx.Client(base_url=base, timeout=10.0)

    # Health
    resp = client.get("/health")
    assert resp.status_code == 200, f"/health returned {resp.status_code}"
    body = resp.json()
    assert body.get("status") == "ok", f"/health body: {body}"
    print(f"  /health OK — model_loaded={body.get('model_loaded')}")

    # Predict (allow 503 gracefully)
    resp = client.post("/predict", json={"features": SAMPLE_FEATURES})
    assert resp.status_code in (200, 503), f"/predict returned {resp.status_code}: {resp.text}"
    print(f"  /predict OK — status={resp.status_code}")

    client.close()
    print("Smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests_inline()
