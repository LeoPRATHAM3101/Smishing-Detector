import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.api import app  # noqa: E402


def test_health_and_predict(small_bundle):
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json={"text": "Update KYC now http://sbi-kyc-update.xyz"})
        assert r.status_code == 200 and r.json()["label"] == "SCAM"


def test_validation(small_bundle):
    with TestClient(app) as c:
        assert c.post("/predict", json={"text": ""}).status_code == 422
        assert c.post("/predict/batch", json={"messages": []}).status_code == 422
        r = c.post("/predict/batch", json={"messages": ["hi", "bit.ly/x KYC blocked"]})
        assert len(r.json()["results"]) == 2
