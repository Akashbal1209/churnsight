"""Policy knowledge base: split markdown policies into section chunks and index them."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from churnsight.config import ARTIFACTS, POLICIES
from churnsight.rag.retriever import HybridIndex, dense_available, encode

EMB_PATH = ARTIFACTS / "policy_embeddings.npy"
CHUNKS_PATH = ARTIFACTS / "policy_chunks.csv"


def load_chunks() -> pd.DataFrame:
    rows = []
    for path in sorted(POLICIES.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        for section in re.split(r"\n(?=## )", text):
            lines = section.strip().splitlines()
            if not lines or not lines[0].startswith("## "):
                continue
            heading = lines[0][3:].strip()
            body = "\n".join(lines[1:]).strip()
            rows.append(
                {
                    "chunk_id": f"{path.stem}#{heading.lower().replace(' ', '-')}",
                    "doc": title,
                    "section": heading,
                    "text": f"{title} - {heading}\n{body}",
                }
            )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def policy_index() -> HybridIndex:
    chunks = load_chunks()
    if EMB_PATH.exists() and CHUNKS_PATH.exists():
        cached = pd.read_csv(CHUNKS_PATH)
        emb = np.load(EMB_PATH)
        if list(cached["chunk_id"]) == list(chunks["chunk_id"]) and list(cached["text"]) == list(chunks["text"]):
            return HybridIndex(chunks, emb, text_col="text", id_col="chunk_id")
    if not dense_available():  # LITE mode: BM25 only, embeddings unused
        return HybridIndex(chunks, np.zeros((len(chunks), 1), np.float32), text_col="text", id_col="chunk_id")
    emb = encode(chunks["text"].tolist())
    np.save(EMB_PATH, emb)
    chunks.to_csv(CHUNKS_PATH, index=False)
    return HybridIndex(chunks, emb, text_col="text", id_col="chunk_id")
