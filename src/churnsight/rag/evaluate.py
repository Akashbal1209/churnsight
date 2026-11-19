"""Retrieval evaluation with exact relevance labels.

The corpus is the 250 gold reviews (LLM-annotated, held out from training), so for every query we know exactly which
documents are relevant (those carrying the query's topic). Each topic gets one English and one
Portuguese query, which also tests cross-lingual retrieval (English question -> Portuguese review).

Output: artifacts/retrieval_eval.csv, artifacts/retrieval_eval.json
Run:  python -m churnsight.rag.evaluate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from churnsight.config import ARTIFACTS, LABELS, PROCESSED
from churnsight.rag.retriever import HybridIndex

QUERIES = {
    "delivery_delay": [
        ("en", "customers complaining the delivery was late or took too long"),
        ("pt", "entrega atrasada, demorou muito para chegar"),
    ],
    "not_received": [
        ("en", "the order never arrived, still waiting for the product"),
        ("pt", "ainda não recebi o produto"),
    ],
    "fast_delivery": [
        ("en", "delivery was fast and arrived before the deadline"),
        ("pt", "chegou antes do prazo, entrega rápida"),
    ],
    "defect_quality": [
        ("en", "product arrived broken, defective or with poor quality"),
        ("pt", "produto com defeito, quebrado, qualidade ruim"),
    ],
    "wrong_missing_item": [
        ("en", "received the wrong item or items were missing from the order"),
        ("pt", "veio produto errado ou faltando itens do pedido"),
    ],
    "good_quality": [
        ("en", "good quality product, exactly as described in the listing"),
        ("pt", "produto de boa qualidade, conforme o anúncio"),
    ],
    "support_refund": [
        ("en", "asked for a refund, return or exchange and the seller did not answer"),
        ("pt", "solicitei devolução ou troca e não tive resposta do atendimento"),
    ],
    "praise_recommend": [
        ("en", "very satisfied customer who recommends the store"),
        ("pt", "recomendo, ótima loja, muito satisfeito"),
    ],
}


def metrics(ranked_ids: list, relevant: set, k: int = 10) -> dict:
    hits = [1 if r in relevant else 0 for r in ranked_ids[:k]]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return {
        "p@5": float(np.mean(hits[:5])),
        "p@10": float(np.mean(hits[:10])),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "ndcg@10": dcg / idcg if idcg else 0.0,
    }


def main() -> None:
    gold = pd.read_csv(LABELS / "gold.csv")
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    emb = np.load(PROCESSED / "review_embeddings.npy")
    pos = pd.Series(np.arange(len(reviews)), index=reviews["review_id"])
    docs = gold.merge(reviews[["review_id", "review_text"]], on="review_id")
    index = HybridIndex(docs, emb[pos.loc[docs["review_id"]].values])

    configs = {
        "BM25 only": dict(mode="bm25"),
        "Dense only": dict(mode="dense"),
        "Hybrid (RRF)": dict(mode="hybrid"),
        "Hybrid + cross-encoder rerank": dict(mode="hybrid", rerank=True),
    }
    rows = []
    for name, cfg in configs.items():
        for topic, qs in QUERIES.items():
            relevant = set(docs.loc[docs[topic] == 1, "review_id"])
            for lang, q in qs:
                res = index.search(q, k=10, **cfg)
                rows.append({"config": name, "topic": topic, "lang": lang, **metrics(list(res["review_id"]), relevant)})
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS / "retrieval_eval_detail.csv", index=False)
    summary = df.groupby("config", sort=False)[["p@5", "p@10", "mrr", "ndcg@10"]].mean().round(3)
    by_lang = df.groupby(["config", "lang"], sort=False)["ndcg@10"].mean().unstack().round(3)
    summary = summary.join(by_lang.add_prefix("ndcg@10_"))
    summary.to_csv(ARTIFACTS / "retrieval_eval.csv")
    (ARTIFACTS / "retrieval_eval.json").write_text(encoding="utf-8", data=
        json.dumps({"corpus": "250 gold reviews (LLM-annotated)", "n_queries": len(df) // len(configs),
                    "summary": summary.reset_index().to_dict(orient="records")}, indent=2)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
