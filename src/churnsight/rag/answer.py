"""Ask ChurnSight: answer analyst questions from customer reviews, with verified citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from churnsight.config import PROCESSED
from churnsight.llm import client
from churnsight.rag.retriever import HybridIndex

SYSTEM = """You are a customer-insights analyst. Answer the question using ONLY the customer reviews
provided (they are in Portuguese; answer in English). Be specific: name the recurring problems, how
often they appear among the provided reviews, and quote short phrases (translated) as evidence.
Cite review ids in `cited_review_ids`. If the reviews do not answer the question, say so and set
confidence to "low". Never use outside knowledge about the company."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_review_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer", "cited_review_ids", "confidence"],
    "additionalProperties": False,
}


@dataclass
class Answer:
    question: str
    reviews: pd.DataFrame
    answer: str | None = None
    cited_review_ids: list[str] = field(default_factory=list)
    confidence: str | None = None
    invalid_citations: list[str] = field(default_factory=list)
    llm_used: bool = False


@lru_cache(maxsize=1)
def review_index() -> HybridIndex:
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    nlp_path = PROCESSED / "review_nlp.parquet"
    if nlp_path.exists():
        reviews = reviews.merge(pd.read_parquet(nlp_path), on="review_id", how="left")
    emb_path = PROCESSED / "review_embeddings.npy"
    # The slim deploy bundle has no embeddings; the index then serves BM25 only (LITE mode).
    emb = np.load(emb_path) if emb_path.exists() else np.zeros((len(reviews), 1), dtype=np.float32)
    return HybridIndex(reviews, emb)


def ask(question: str, filters: dict | None = None, k: int = 12, rerank: bool = False,
        use_llm: bool | None = None) -> Answer:
    hits = review_index().search(question, k=k, filters=filters, mode="hybrid", rerank=rerank)
    out = Answer(question, hits)
    if use_llm is None:
        use_llm = client.available()
    if not use_llm or hits.empty:
        return out
    context = "\n".join(
        f"[{r.review_id}] ({int(r.review_score)} stars, {r.order_month}, {r.category}) {r.review_text}"
        for r in hits.itertuples()
    )
    res = client.complete_json(SYSTEM, f"REVIEWS:\n{context}\n\nQUESTION: {question}", SCHEMA, max_tokens=4000)
    allowed = set(hits["review_id"])
    out.answer, out.confidence, out.llm_used = res["answer"], res["confidence"], True
    out.cited_review_ids = [c for c in res["cited_review_ids"] if c in allowed]
    out.invalid_citations = [c for c in res["cited_review_ids"] if c not in allowed]
    return out
