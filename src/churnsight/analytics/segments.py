"""Customer segmentation: RFM + experience features -> KMeans -> human-readable segment names.

Olist is dominated by one-time buyers, so plain RFM collapses (F is almost always 1).
We add *experience* features (review score, late delivery) so segments separate
"happy one-timers" from "unhappy one-timers", which matters for retention action.

Outputs:
  data/processed/customer_segments.parquet
  artifacts/segment_profiles.csv, artifacts/segmentation_metrics.json

Run:  python -m churnsight.analytics.segments
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from churnsight.config import ARTIFACTS, PROCESSED, RANDOM_SEED

FEATURES = ["recency_days", "frequency", "monetary", "avg_review_score", "late_share"]


def build_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    snapshot = orders["purchase_date"].max() + pd.Timedelta(days=1)
    g = orders.groupby("customer_unique_id")
    rfm = pd.DataFrame(
        {
            "recency_days": (snapshot - g["purchase_date"].max()).dt.days,
            "frequency": g.size(),
            "monetary": g["payment_value"].sum(),
            "avg_review_score": g["review_score"].mean(),
            "late_share": g["is_late"].mean(),
            "customer_state": g["customer_state"].first(),
        }
    )
    rfm["avg_review_score"] = rfm["avg_review_score"].fillna(rfm["avg_review_score"].median())
    return rfm


def _matrix(rfm: pd.DataFrame) -> np.ndarray:
    x = rfm[FEATURES].copy()
    x["monetary"] = np.log1p(x["monetary"])
    x["frequency"] = np.log1p(x["frequency"])
    return StandardScaler().fit_transform(x)


def choose_k(x: np.ndarray, ks=range(3, 9), min_k: int = 5) -> tuple[int, dict]:
    """Silhouette alone prefers k=3, but that puts ~80% of customers in one segment, which no
    team can act on. We report silhouette for every k and pick the best k >= min_k."""
    rng = np.random.default_rng(RANDOM_SEED)
    sample = x[rng.choice(len(x), size=min(20000, len(x)), replace=False)]
    scores = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(sample)
        scores[k] = float(silhouette_score(sample, km.labels_))
    eligible = {k: s for k, s in scores.items() if k >= min_k}
    return max(eligible, key=eligible.get), scores


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Deterministic names from centroid profiles, so names are explainable."""
    overall = profile.median()
    names = {}
    for seg, p in profile.iterrows():
        if p["frequency"] >= 1.5:
            names[seg] = "Loyal Repeat Buyers"
        elif p["avg_review_score"] <= 2.5:
            names[seg] = "Unhappy One-timers"
        elif p["late_share"] >= 0.5:
            names[seg] = "Burned by Late Delivery"
        elif p["monetary"] >= 2 * overall["monetary"]:
            names[seg] = "High-Value One-timers"
        elif p["recency_days"] <= 0.7 * overall["recency_days"]:
            names[seg] = "Recent Happy Buyers"
        else:
            names[seg] = "Lapsed Happy Buyers"
    # Disambiguate duplicates
    seen: dict[str, int] = {}
    for seg in sorted(names):
        n = names[seg]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[seg] = f"{n} ({seen[n]})"
    return names


def main() -> None:
    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    rfm = build_rfm(orders)
    x = _matrix(rfm)
    k, sil = choose_k(x)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(x)
    rfm["segment_id"] = km.labels_

    profile = rfm.groupby("segment_id")[FEATURES].mean()
    profile["size"] = rfm.groupby("segment_id").size()
    profile["share"] = (profile["size"] / len(rfm)).round(4)
    names = name_segments(profile)
    profile["segment"] = profile.index.map(names)
    rfm["segment"] = rfm["segment_id"].map(names)

    rfm.reset_index().to_parquet(PROCESSED / "customer_segments.parquet", index=False)
    profile.round(3).to_csv(ARTIFACTS / "segment_profiles.csv")
    (ARTIFACTS / "segmentation_metrics.json").write_text(encoding="utf-8", data=
        json.dumps({"chosen_k": k, "silhouette_by_k": sil}, indent=2)
    )
    print(profile.round(2).to_string())
    print("k =", k, "silhouette:", {kk: round(v, 3) for kk, v in sil.items()})


if __name__ == "__main__":
    main()
