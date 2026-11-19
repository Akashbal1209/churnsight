"""Cohort retention + EDA summary.

Outputs (artifacts/):
  cohort_retention.csv      rows = first-purchase month, cols = months since first purchase (% returning)
  eda_summary.json          headline numbers used in the README and dashboard
  figures/*.png             charts for the README

Run:  python -m churnsight.analytics.cohorts
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from churnsight.config import ARTIFACTS, PROCESSED

FIG = ARTIFACTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def cohort_matrix(orders: pd.DataFrame, max_months: int = 12) -> pd.DataFrame:
    o = orders[["customer_unique_id", "purchase_date"]].copy()
    o["month"] = o["purchase_date"].dt.to_period("M")
    o["cohort"] = o.groupby("customer_unique_id")["month"].transform("min")
    o["k"] = (o["month"] - o["cohort"]).apply(lambda x: x.n)
    sizes = o[o["k"] == 0].groupby("cohort")["customer_unique_id"].nunique()
    active = o[o["k"] > 0].groupby(["cohort", "k"])["customer_unique_id"].nunique().unstack(fill_value=0)
    m = active.reindex(index=sizes.index, columns=range(1, max_months + 1), fill_value=0)
    m = m.div(sizes, axis=0) * 100
    m.insert(0, "cohort_size", sizes)
    # Keep cohorts with a meaningful size (Olist is sparse before 2017).
    m = m[m["cohort_size"] >= 500]
    # Blank out cells that are in the future relative to the data end.
    last = o["month"].max()
    for cohort in m.index:
        for k in range(1, max_months + 1):
            if cohort + k > last:
                m.loc[cohort, k] = float("nan")
    m.index = m.index.astype(str)
    return m.round(2)


def plot_cohort(m: pd.DataFrame) -> None:
    data = m.drop(columns="cohort_size")
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(data.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(data.shape[1]), labels=[str(c) for c in data.columns])
    ax.set_yticks(range(data.shape[0]), labels=data.index)
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("First-purchase cohort")
    ax.set_title("Monthly retention by cohort (% of cohort ordering again)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(FIG / "cohort_retention.png", dpi=130)
    plt.close(fig)


def eda_summary(orders: pd.DataFrame, customers: pd.DataFrame, reviews: pd.DataFrame) -> dict:
    obs = customers[customers["observable"]]
    by_late = obs.groupby("is_late")["repurchased"].mean()
    by_score = obs.groupby("review_score")["repurchased"].mean()
    return {
        "n_orders": int(len(orders)),
        "n_customers": int(orders["customer_unique_id"].nunique()),
        "share_one_time_customers": round(float((customers["n_orders_total"] == 1).mean()), 4),
        "repurchase_rate_180d": round(float(obs["repurchased"].mean()), 4),
        "late_delivery_rate": round(float(orders["is_late"].mean()), 4),
        "avg_review_score": round(float(orders["review_score"].mean()), 3),
        "share_reviews_with_text": round(float(orders["has_review_text"].mean()), 4),
        "n_text_reviews_clean": int(len(reviews)),
        "repurchase_rate_on_time": round(float(by_late.get(0, float("nan"))), 4),
        "repurchase_rate_late": round(float(by_late.get(1, float("nan"))), 4),
        "avg_review_score_on_time": round(float(orders.loc[orders.is_late == 0, "review_score"].mean()), 3),
        "avg_review_score_late": round(float(orders.loc[orders.is_late == 1, "review_score"].mean()), 3),
        "repurchase_rate_by_review_score": {str(int(k)): round(float(v), 4) for k, v in by_score.items()},
    }


def plot_basics(orders: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    monthly = orders.groupby("order_month").size()
    monthly = monthly[monthly.index >= "2017-01"]
    axes[0].plot(monthly.index, monthly.values, marker="o")
    axes[0].set_title("Delivered orders per month")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)

    orders["review_score"].value_counts().sort_index().plot.bar(ax=axes[1], color="#4C72B0")
    axes[1].set_title("Review score distribution")

    late = orders.groupby("is_late")["review_score"].mean()
    axes[2].bar(["On time", "Late"], late.values, color=["#55A868", "#C44E52"])
    axes[2].set_title("Average review score: on-time vs late")
    axes[2].set_ylim(0, 5)
    fig.tight_layout()
    fig.savefig(FIG / "eda_basics.png", dpi=130)
    plt.close(fig)


def main() -> None:
    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    customers = pd.read_parquet(PROCESSED / "customers.parquet")
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")

    m = cohort_matrix(orders)
    m.to_csv(ARTIFACTS / "cohort_retention.csv")
    plot_cohort(m)
    plot_basics(orders)

    summary = eda_summary(orders, customers, reviews)
    (ARTIFACTS / "eda_summary.json").write_text(encoding="utf-8", data=json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
