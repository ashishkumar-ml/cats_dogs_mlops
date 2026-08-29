"""
Unit tests for api/predictor.py and the FastAPI endpoints.
"""
import io
import os
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from api.main import app
from api.predictor import Predictor, CLASS_NAMES
from src.preprocess import preprocess_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(w=64, h=64) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Predictor unit tests
# ---------------------------------------------------------------------------

class TestPredictor:
    def test_loads_without_weights_file(self, tmp_path):
        """Predictor must load even when no saved weights exist (uses random init)."""
        p = Predictor(model_path=str(tmp_path / "nonexistent.pt"))
        p.load()
        assert p.is_loaded

    def test_predict_returns_known_class(self, tmp_path):
        p = Predictor(model_path=str(tmp_path / "nonexistent.pt"))
        p.load()
        image_bytes = _make_jpeg_bytes()
        cls, conf, probs = p.predict(image_bytes)
        assert cls in CLASS_NAMES
        assert 0.0 <= conf <= 1.0

    def test_predict_probabilities_sum_to_one(self, tmp_path):
        p = Predictor(model_path=str(tmp_path / "nonexistent.pt"))
        p.load()
        _, _, probs = p.predict(_make_jpeg_bytes())
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}"

    def test_predict_all_classes_present(self, tmp_path):
        p = Predictor(model_path=str(tmp_path / "nonexistent.pt"))
        p.load()
        _, _, probs = p.predict(_make_jpeg_bytes())
        for cls in CLASS_NAMES:
            assert cls in probs

    def test_predict_confidence_matches_max_prob(self, tmp_path):
        p = Predictor(model_path=str(tmp_path / "nonexistent.pt"))
        p.load()
        cls, conf, probs = p.predict(_make_jpeg_bytes())
        assert abs(probs[cls] - conf) < 1e-4

    def test_lazy_load_on_predict(self, tmp_path):
        """Predictor should load itself on first predict call."""
        p = Predictor(model_path=str(tmp_path / "nonexistent.pt"))
        assert not p.is_loaded
        p.predict(_make_jpeg_bytes())
        assert p.is_loaded


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_response_has_version(self, client):
        data = client.get("/health").json()
        assert "version" in data

    def test_model_loaded_flag(self, client):
        data = client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    def _post_image(self, client, image_bytes, content_type="image/jpeg"):
        return client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, content_type)},
        )

    def test_valid_jpeg_returns_200(self, client):
        resp = self._post_image(client, _make_jpeg_bytes())
        assert resp.status_code == 200

    def test_response_contains_predicted_class(self, client):
        data = self._post_image(client, _make_jpeg_bytes()).json()
        assert data["predicted_class"] in CLASS_NAMES

    def test_response_confidence_in_range(self, client):
        data = self._post_image(client, _make_jpeg_bytes()).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_probabilities_present(self, client):
        data = self._post_image(client, _make_jpeg_bytes()).json()
        assert "probabilities" in data
        for cls in CLASS_NAMES:
            assert cls in data["probabilities"]

    def test_invalid_content_type_returns_400(self, client):
        resp = self._post_image(client, b"not-an-image", content_type="text/plain")
        assert resp.status_code == 400

    def test_metrics_endpoint_reachable(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "inference_requests_total" in resp.text
