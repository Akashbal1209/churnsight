"""Embeddings for every review + BERTopic clustering of complaints (1-2 star reviews).

The fixed topic classifier answers "which known problem is this?". Clustering answers
"which recurring problems exist that we did not define upfront?".

Outputs:
  data/processed/review_embeddings.npy   (aligned with reviews.parquet rows; reused by RAG)
  data/processed/complaint_clusters.parquet  review_id -> cluster
  artifacts/complaint_clusters.json      per cluster: size, keywords, representative reviews,
                                          avg score, late share, top categories

Run:  python -m churnsight.nlp.clusters
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from churnsight.config import ARTIFACTS, PROCESSED, RANDOM_SEED

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PT_STOPWORDS = """a ao aos aquela aquele aqueles as at até com como da das de dela dele deles depois
do dos e ela ele eles em entre era essa esse esta este eu foi for foram fui há isso isto já la lhe
mais mas me mesmo meu minha muito na nas nem no nos nossa nosso num numa o os ou para pela pelas
pelo pelos por qual quando que quem se sem ser seu sua são só também te tem ter teu tinha um uma
você vocês vc vcs pra pro ainda até então porém pois q ja nao não sim estou está estava foi seria
ate tb tambem produto produtos compra comprei loja""".split()


def embed_all(reviews: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        reviews["review_text"].tolist(), batch_size=128, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def main() -> None:
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    emb_path = PROCESSED / "review_embeddings.npy"
    if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == len(reviews):
        emb = np.load(emb_path)
    else:
        emb = embed_all(reviews)
        np.save(emb_path, emb)

    mask = (reviews["review_score"] <= 2).values
    neg = reviews[mask].reset_index(drop=True)
    neg_emb = emb[mask]

    topic_model = BERTopic(
        embedding_model=SentenceTransformer(EMBED_MODEL),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=RANDOM_SEED),
        # "eom" selection collapses short complaints into one giant cluster (>90%); "leaf" gives
        # ~28 specific clusters but leaves ~45% outliers, which we then reassign by similarity.
        hdbscan_model=HDBSCAN(min_cluster_size=80, min_samples=10, metric="euclidean",
                              cluster_selection_method="leaf"),
        vectorizer_model=CountVectorizer(stop_words=PT_STOPWORDS, ngram_range=(1, 2), min_df=5),
        calculate_probabilities=False,
        verbose=True,
    )
    docs = neg["review_text"].tolist()
    topics, _ = topic_model.fit_transform(docs, neg_emb)
    raw_outliers = float(np.mean(np.array(topics) == -1))
    topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=neg_emb, threshold=0.5)
    topic_model.update_topics(docs, topics=topics, vectorizer_model=CountVectorizer(
        stop_words=PT_STOPWORDS, ngram_range=(1, 2), min_df=5))
    neg["cluster"] = topics
    neg[["review_id", "cluster"]].to_parquet(PROCESSED / "complaint_clusters.parquet", index=False)

    info = topic_model.get_topic_info()
    out = []
    for _, row in info.iterrows():
        cid = int(row["Topic"])
        if cid == -1:
            continue
        sub = neg[neg["cluster"] == cid]
        out.append(
            {
                "cluster": cid,
                "size": int(len(sub)),
                "keywords": [w for w, _ in topic_model.get_topic(cid)[:10]],
                "representative_reviews": list(row["Representative_Docs"])[:5],
                "avg_review_score": round(float(sub["review_score"].mean()), 2),
                "late_share": round(float(sub["is_late"].mean()), 3),
                "top_categories": sub["category"].value_counts().head(3).to_dict(),
                "first_month": str(sub["order_month"].min()),
            }
        )
    meta = {
        "embedding_model": EMBED_MODEL,
        "n_complaints": int(len(neg)),
        "n_clusters": len(out),
        "outlier_share_before_reassignment": round(raw_outliers, 3),
        "outlier_share": round(float((neg["cluster"] == -1).mean()), 3),
        "clusters": out,
    }
    (ARTIFACTS / "complaint_clusters.json").write_text(encoding="utf-8", data=json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{len(neg)} complaints -> {len(out)} clusters, outliers {meta['outlier_share']:.1%}")
    for c in out:
        print(c["cluster"], c["size"], ", ".join(c["keywords"][:6]))


if __name__ == "__main__":
    main()
