"""ChurnSight REST API.  Run:  uvicorn app.api:app --reload"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnsight.config import ARTIFACTS  # noqa: E402
from churnsight.llm import client  # noqa: E402
from churnsight.llm.actions import Case, suggest  # noqa: E402
from churnsight.nlp.weak_labels import TOPICS  # noqa: E402

app = FastAPI(title="ChurnSight API", version="0.1.0",
              description="Customer review intelligence + retention analytics on the Olist dataset.")


def _artifact(name: str):
    p = ARTIFACTS / name
    if not p.exists():
        raise HTTPException(404, f"{name} not generated yet; run the pipeline")
    return json.loads(p.read_text(encoding="utf-8"))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Why do customers complain about missing items?"])
    min_stars: int = Field(1, ge=1, le=5)
    max_stars: int = Field(5, ge=1, le=5)
    from_month: str | None = Field(None, examples=["2017-06"])
    k: int = Field(10, ge=1, le=30)


class CaseRequest(BaseModel):
    review_text: str
    review_score: int = Field(..., ge=1, le=5)
    payment_value: float = Field(..., ge=0)
    delay_days: float | None = None
    topics: list[str] = Field(default_factory=list, description=f"subset of {list(TOPICS)}")


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": client.available(), "model": client.model() if client.available() else None}


@app.get("/insights/summary")
def summary():
    return _artifact("eda_summary.json")


@app.get("/insights/alerts")
def alerts():
    return _artifact("alerts.json")


@app.get("/insights/clusters")
def clusters():
    p = ARTIFACTS / "cluster_insights.json"
    return _artifact("cluster_insights.json" if p.exists() else "complaint_clusters.json")


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    from churnsight.rag.answer import ask

    filters = {"review_score": {"min": req.min_stars, "max": req.max_stars}}
    if req.from_month:
        filters["order_month"] = {"min": req.from_month}
    res = ask(req.question, filters=filters, k=req.k)
    return {
        "answer": res.answer,
        "confidence": res.confidence,
        "cited_review_ids": res.cited_review_ids,
        "llm_used": res.llm_used,
        "reviews": res.reviews[["review_id", "review_score", "order_month", "category", "review_text"]].to_dict(orient="records"),
    }


@app.post("/suggest-reply")
def suggest_reply(req: CaseRequest):
    unknown = [t for t in req.topics if t not in TOPICS]
    if unknown:
        raise HTTPException(422, f"unknown topics: {unknown}")
    case = Case(review_id="api", review_text=req.review_text, review_score=req.review_score,
                topics=req.topics, payment_value=req.payment_value, delay_days=req.delay_days)
    return suggest(case).to_dict()
