"""API smoke tests in LITE mode (no neural models, no LLM key needed)."""

import os

os.environ["CHURNSIGHT_LITE"] = "1"
os.environ["LLM_PROVIDER"] = "none"

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["llm_enabled"] is False


def test_suggest_reply_rules_without_llm():
    r = client.post("/suggest-reply", json={
        "review_text": "Comprei 2 unidades e veio apenas uma, já pedi o reembolso e ninguém responde",
        "review_score": 1, "payment_value": 120.0, "delay_days": None,
        "topics": ["wrong_missing_item", "support_refund"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["priority"] == "high"
    assert body["eligible_voucher_brl"] == 15
    assert body["llm_used"] is False
    ids = [c["chunk_id"] for c in body["policy_chunks"]]
    assert any(i.startswith("05-wrong-missing-items") for i in ids)
    assert "06-retention-goodwill#allowed-offers" in ids


def test_unknown_topic_rejected():
    r = client.post("/suggest-reply", json={"review_text": "x", "review_score": 1, "payment_value": 1,
                                            "topics": ["nonsense"]})
    assert r.status_code == 422
