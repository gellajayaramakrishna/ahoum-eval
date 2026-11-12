import time

import pytest

import app as app_module
import scorer
from conftest import SAMPLE
from fakes import FakeGroq


@pytest.fixture
def client():
    return app_module.app.test_client()


def test_health(client):
    h = client.get("/health").json
    assert h["ok"] and h["lite_model"] and h["facets"] == 399


def test_validation(client):
    assert client.post("/score", json={"conversation": ""}).status_code == 400
    assert client.post("/score", json={"conversation": "x", "mode": "nope"}).status_code == 400
    assert client.post("/score", json={"conversation": "x" * 30000}).status_code == 413


def test_lite_sync(client):
    r = client.post("/api/score", json={"conversation": SAMPLE, "mode": "lite"}).json
    assert r["total"] == 399 and r["info"]["mode"] == "lite"


def test_async_job_completes(client):
    job = client.post("/score", json={"conversation": SAMPLE, "mode": "lite"}).json
    for _ in range(50):
        s = client.get(f"/status/{job['job_id']}").json
        if s["status"] != "running":
            break
        time.sleep(0.1)
    assert s["status"] == "complete" and s["result"]["total"] == 399


def test_hybrid_routes_uncertain_facets_to_llm(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    scorer._cache.clear()
    fake = FakeGroq()
    scorer.set_client(fake)
    try:
        r = client.post("/api/score", json={"conversation": SAMPLE, "mode": "hybrid", "llm_share": 0.2}).json
    finally:
        scorer.set_client(None)
    src = [v["source"] for v in r["scores"].values()]
    assert src.count("llm") == round(399 * 0.2) and src.count("lite") == 399 - round(399 * 0.2)
    assert r["info"]["routed"] == 80


def test_hybrid_without_key_falls_back(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = client.post("/api/score", json={"conversation": SAMPLE, "mode": "hybrid"}).json
    assert r["info"]["mode"] == "lite" and r["info"]["notes"]


def test_metrics_endpoint(client):
    m = client.get("/metrics").json
    assert {"teacher_audit", "lite_metrics", "emotion_metrics"} <= set(m)
