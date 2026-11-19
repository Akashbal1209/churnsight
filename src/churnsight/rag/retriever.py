"""Hybrid retrieval over reviews (and policy docs): BM25 + dense embeddings, RRF fusion,
optional cross-encoder rerank, metadata filters.

Vector store: exact cosine search in NumPy. With ~34k x 384-dim vectors a brute-force search
takes a few milliseconds, so an ANN index (Qdrant/FAISS) would add complexity without benefit
at this scale; the interface is small enough to swap one in later.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from unidecode import unidecode

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RRF_K = 60

_encoder = None
_reranker = None


def dense_available() -> bool:
    """False in LITE mode (CHURNSIGHT_LITE=1, for <1 GB RAM hosts) or without sentence-transformers."""
    if os.getenv("CHURNSIGHT_LITE", "0") == "1":
        return False
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", unidecode(str(text)).lower()) if len(t) > 1]


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(EMBED_MODEL)
    return _encoder


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL, max_length=256)
    return _reranker


def encode(texts: list[str]) -> np.ndarray:
    return get_encoder().encode(texts, normalize_embeddings=True, batch_size=64).astype(np.float32)


@dataclass
class HybridIndex:
    docs: pd.DataFrame  # must contain `text_col` and `id_col`
    embeddings: np.ndarray  # L2-normalised, aligned with docs rows
    text_col: str = "review_text"
    id_col: str = "review_id"
    _bm25: BM25Okapi | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.docs = self.docs.reset_index(drop=True)
        self._bm25 = BM25Okapi([tokenize(t) for t in self.docs[self.text_col]])

    def _mask(self, filters: dict | None) -> np.ndarray:
        m = np.ones(len(self.docs), dtype=bool)
        for col, val in (filters or {}).items():
            if val is None:
                continue
            s = self.docs[col]
            if isinstance(val, (list, tuple, set)):
                m &= s.isin(list(val)).values
            elif isinstance(val, dict):  # range filter {"min": .., "max": ..}
                if "min" in val:
                    m &= (s >= val["min"]).values
                if "max" in val:
                    m &= (s <= val["max"]).values
            else:
                m &= (s == val).values
        return m

    def search(
        self, query: str, k: int = 10, filters: dict | None = None, mode: str = "hybrid",
        rerank: bool = False, candidates: int = 50,
    ) -> pd.DataFrame:
        if not dense_available():
            mode, rerank = "bm25", False
        mask = self._mask(filters)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return self.docs.iloc[[]].assign(score=[])

        rankings = []
        if mode in ("dense", "hybrid"):
            q = encode([query])[0]
            dense = self.embeddings[idx] @ q
            rankings.append(idx[np.argsort(-dense)[:candidates]])
        if mode in ("bm25", "hybrid"):
            bm = self._bm25.get_scores(tokenize(query))[idx]
            order = np.argsort(-bm)[:candidates]
            rankings.append(idx[order[bm[order] > 0]])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + r + 1)
        cand = sorted(fused, key=fused.get, reverse=True)[:candidates]
        scores = [fused[i] for i in cand]

        if rerank and cand:
            pairs = [(query, self.docs.at[i, self.text_col]) for i in cand]
            scores = list(get_reranker().predict(pairs))
            order = np.argsort(scores)[::-1]
            cand, scores = [cand[i] for i in order], [float(scores[i]) for i in order]

        out = self.docs.iloc[cand[:k]].copy()
        out["score"] = scores[:k]
        return out
