"""Issue impact (complaint -> repurchase -> revenue) and complaint trend alerts.

Joins what customers SAID in their first review (topic probabilities from the fine-tuned model)
with what they DID next (repurchase in the window), and turns the gap into money.

Outputs (artifacts/):
  issue_impact.csv      per topic: customers, repurchase rate vs other reviewers, 95% CI, revenue at stake
  seller_hotspots.csv   sellers with the highest complaint rates (min volume)
  topic_trends.csv      monthly share of reviews mentioning each topic
  alerts.json           topics whose latest month is a statistical spike (z-score)

Run:  python -m churnsight.analytics.impact
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from churnsight.config import ARTIFACTS, PREDICTION_DAY, PROCESSED, RANDOM_SEED, REPURCHASE_WINDOW_DAYS
from churnsight.nlp.weak_labels import TOPICS, topic_thresholds




def _topic_flags(nlp: pd.DataFrame) -> pd.DataFrame:
    flags = nlp[["review_id"]].copy()
    th = topic_thresholds()
    for t in TOPICS:
        flags[t] = (nlp[f"topic_{t}"] >= th[t]).astype(int)
    return flags


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def future_value_per_returner(orders: pd.DataFrame, customers: pd.DataFrame) -> float:
    """Average revenue a returning customer brings in the window after the prediction day."""
    first = customers.set_index("customer_unique_id")["purchase_date"]
    o = orders.assign(first_date=orders["customer_unique_id"].map(first))
    d = (o["purchase_date"] - o["first_date"]).dt.total_seconds() / 86400
    w = o[(d > PREDICTION_DAY) & (d <= PREDICTION_DAY + REPURCHASE_WINDOW_DAYS)]
    per_customer = w.groupby("customer_unique_id")["payment_value"].sum()
    return float(per_customer.mean())


def issue_impact(customers, flags, orders) -> pd.DataFrame:
    obs = customers[customers["observable"] & customers["review_id"].notna()]
    obs = obs.merge(flags, on="review_id", how="inner")
    all_reviewed = customers[customers["review_id"].notna()].merge(flags, on="review_id", how="inner")
    value = future_value_per_returner(orders, customers)
    rows = []
    for t, label in TOPICS.items():
        y_t = obs.loc[obs[t] == 1, "repurchased"].values
        y_o = obs.loc[obs[t] == 0, "repurchased"].values
        if len(y_t) < 30:
            continue
        lo, hi = bootstrap_diff(y_t, y_o)
        diff = y_t.mean() - y_o.mean()
        n_all = int(all_reviewed[t].sum())
        rows.append(
            {
                "topic": t,
                "label": label,
                "customers_observed": len(y_t),
                "repurchase_rate": round(float(y_t.mean()), 4),
                "repurchase_rate_others": round(float(y_o.mean()), 4),
                "diff_pp": round(diff * 100, 2),
                "ci95_low_pp": round(lo * 100, 2),
                "ci95_high_pp": round(hi * 100, 2),
                "significant": bool(lo > 0 or hi < 0),
                "relative_risk": round(float(y_t.mean() / max(y_o.mean(), 1e-9)), 3),
                "customers_with_topic_all": n_all,
                "revenue_at_stake_brl": round(-diff * n_all * value, 0),
            }
        )
    out = pd.DataFrame(rows).sort_values("diff_pp")
    out.attrs["value_per_returner_brl"] = value
    return out


def seller_hotspots(reviews: pd.DataFrame, flags: pd.DataFrame, orders: pd.DataFrame, min_reviews: int = 50):
    r = reviews.merge(flags, on="review_id")
    r["any_complaint"] = r[["delivery_delay", "not_received", "defect_quality", "wrong_missing_item"]].max(axis=1)
    g = r.groupby("seller_id").agg(
        text_reviews=("review_id", "count"),
        complaint_rate=("any_complaint", "mean"),
        delay_rate=("delivery_delay", "mean"),
        defect_rate=("defect_quality", "mean"),
        wrong_item_rate=("wrong_missing_item", "mean"),
        avg_score=("review_score", "mean"),
    )
    g = g[g["text_reviews"] >= min_reviews]
    g["orders"] = orders.groupby("seller_id").size().reindex(g.index)
    g["main_category"] = orders.groupby("seller_id")["category"].agg(lambda s: s.mode().iat[0]).reindex(g.index)
    platform = float(r["any_complaint"].mean())
    g["complaint_rate_vs_platform"] = (g["complaint_rate"] / platform).round(2)
    return g.sort_values("complaint_rate", ascending=False).round(3).head(25)


def trends_and_alerts(reviews: pd.DataFrame, flags: pd.DataFrame, lookback: int = 6, z_thresh: float = 2.0):
    r = reviews.merge(flags, on="review_id")
    r = r[(r["order_month"] >= "2017-01") & (r["order_month"] <= "2018-08")]
    monthly = r.groupby("order_month")[list(TOPICS)].mean()
    monthly["n_reviews"] = r.groupby("order_month").size()
    complaints = ["delivery_delay", "not_received", "defect_quality", "wrong_missing_item", "support_refund"]
    alerts = detect_spikes(monthly[complaints], lookback, z_thresh)
    # Backtest: what would the weekly report have flagged at the end of each past month?
    history = []
    for i in range(lookback + 1, len(monthly) + 1):
        history += detect_spikes(monthly[complaints].iloc[:i], lookback, z_thresh)
    return monthly.round(4), alerts, history


def detect_spikes(monthly: pd.DataFrame, lookback: int = 6, z_thresh: float = 2.0) -> list[dict]:
    """Flag topics whose latest month share is > z_thresh std devs above the trailing mean."""
    alerts = []
    if len(monthly) <= lookback:
        return alerts
    last = monthly.index[-1]
    hist = monthly.iloc[-lookback - 1 : -1]
    for t in monthly.columns:
        mu, sd = hist[t].mean(), hist[t].std(ddof=1)
        cur = monthly[t].iloc[-1]
        z = (cur - mu) / sd if sd > 0 else 0.0
        if z >= z_thresh:
            alerts.append(
                {"topic": t, "label": TOPICS.get(t, t), "month": str(last), "share": round(float(cur), 4),
                 "baseline_share": round(float(mu), 4), "z_score": round(float(z), 2),
                 "change_pct": round(float((cur / mu - 1) * 100), 1) if mu > 0 else None}
            )
    return sorted(alerts, key=lambda a: -a["z_score"])


def main() -> None:
    customers = pd.read_parquet(PROCESSED / "customers.parquet")
    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    flags = _topic_flags(pd.read_parquet(PROCESSED / "review_nlp.parquet"))

    imp = issue_impact(customers, flags, orders)
    imp.to_csv(ARTIFACTS / "issue_impact.csv", index=False)
    hot = seller_hotspots(reviews, flags, orders)
    hot.to_csv(ARTIFACTS / "seller_hotspots.csv")
    monthly, alerts, history = trends_and_alerts(reviews, flags)
    monthly.to_csv(ARTIFACTS / "topic_trends.csv")
    (ARTIFACTS / "alerts.json").write_text(encoding="utf-8", data=json.dumps(alerts, indent=2))
    (ARTIFACTS / "alerts_history.json").write_text(encoding="utf-8", data=json.dumps(history, indent=2))
    print("\nBacktest alerts:", json.dumps(history, indent=1))

    print("value per returning customer (BRL):", round(imp.attrs["value_per_returner_brl"], 2))
    print(imp.to_string(index=False))
    print("\nTop seller hotspots:\n", hot.head(8).to_string())
    print("\nAlerts:", json.dumps(alerts, indent=2))


if __name__ == "__main__":
    main()
