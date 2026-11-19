"""Join the raw Olist tables into clean, analysis-ready tables.

Outputs (data/processed/):
  orders.parquet     one row per delivered order: customer, money, delivery, review
  reviews.parquet    one row per review that has written text (input to NLP)
  customers.parquet  one row per real customer (customer_unique_id): first-order experience as
                     known PREDICTION_DAY days after purchase + whether they came back in the
                     following REPURCHASE_WINDOW_DAYS (the target)

Run:  python -m churnsight.data.build
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from churnsight.config import ARTIFACTS, PREDICTION_DAY, PROCESSED, RAW, REPURCHASE_WINDOW_DAYS

SRC = RAW / "olist"


def _read(name: str, dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(SRC / f"olist_{name}_dataset.csv", parse_dates=dates or [])


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "orders": _read(
            "orders",
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "customers": _read("customers"),
        "items": _read("order_items", ["shipping_limit_date"]),
        "payments": _read("order_payments"),
        "reviews": _read("order_reviews", ["review_creation_date", "review_answer_timestamp"]),
        "products": _read("products"),
        "sellers": _read("sellers"),
        "category_en": pd.read_csv(SRC / "product_category_name_translation.csv"),
    }


def clean_text(s: object) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_orders(t: dict[str, pd.DataFrame], log: dict) -> pd.DataFrame:
    orders = t["orders"]
    log["orders_raw"] = len(orders)
    orders = orders[orders["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date"]
    )
    log["orders_delivered"] = len(orders)

    # Items -> order level. Main category/seller = the most expensive item in the order.
    items = t["items"].merge(t["products"][["product_id", "product_category_name"]], how="left")
    items = items.merge(t["category_en"], how="left")
    items["category"] = items["product_category_name_english"].fillna(
        items["product_category_name"]
    ).fillna("unknown")
    items = items.merge(
        t["sellers"][["seller_id", "seller_state"]], how="left", on="seller_id"
    )
    main_item = items.sort_values("price", ascending=False).drop_duplicates("order_id")
    item_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        items_price=("price", "sum"),
        freight=("freight_value", "sum"),
    )
    item_agg = item_agg.join(
        main_item.set_index("order_id")[["category", "seller_id", "seller_state"]]
    )

    pay = t["payments"]
    main_pay = pay.sort_values("payment_value", ascending=False).drop_duplicates("order_id")
    pay_agg = pay.groupby("order_id").agg(
        payment_value=("payment_value", "sum"), installments=("payment_installments", "max")
    )
    pay_agg = pay_agg.join(main_pay.set_index("order_id")[["payment_type"]])

    # Some orders have >1 review: keep the latest one.
    rev = t["reviews"].sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    rev = rev.assign(
        review_title=rev["review_comment_title"].map(clean_text),
        review_message=rev["review_comment_message"].map(clean_text),
    )
    rev["review_text"] = (rev["review_title"] + ". " + rev["review_message"]).str.strip(". ").str.strip()
    rev = rev[["order_id", "review_id", "review_score", "review_creation_date", "review_text"]]

    df = (
        orders.merge(t["customers"], on="customer_id", how="left")
        .merge(item_agg, left_on="order_id", right_index=True, how="inner")
        .merge(pay_agg, left_on="order_id", right_index=True, how="left")
        .merge(rev, on="order_id", how="left")
    )
    log["orders_with_items"] = len(df)

    d = df
    d["purchase_date"] = d["order_purchase_timestamp"]
    d["delivery_days"] = (d["order_delivered_customer_date"] - d["purchase_date"]).dt.total_seconds() / 86400
    d["promised_days"] = (d["order_estimated_delivery_date"] - d["purchase_date"]).dt.total_seconds() / 86400
    d["delay_days"] = (
        d["order_delivered_customer_date"] - d["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    d["is_late"] = (d["delay_days"] > 0).astype(int)
    d["freight_ratio"] = d["freight"] / d["items_price"].replace(0, np.nan)
    d["same_state_seller"] = (d["seller_state"] == d["customer_state"]).astype(int)
    d["has_review_text"] = d["review_text"].fillna("").str.len().gt(0).astype(int)
    d["order_month"] = d["purchase_date"].dt.to_period("M").astype(str)

    keep = [
        "order_id", "customer_unique_id", "customer_state", "customer_city", "purchase_date",
        "order_month", "n_items", "n_sellers", "items_price", "freight", "freight_ratio",
        "payment_value", "installments", "payment_type", "category", "seller_id", "seller_state",
        "same_state_seller", "delivery_days", "promised_days", "delay_days", "is_late",
        "review_id", "review_score", "review_creation_date", "review_text", "has_review_text",
    ]
    return d[keep].sort_values("purchase_date").reset_index(drop=True)


def build_reviews(orders: pd.DataFrame, log: dict) -> pd.DataFrame:
    r = orders[orders["has_review_text"] == 1].copy()
    r = r.drop_duplicates("review_text")  # identical boilerplate texts add no information
    r["n_words"] = r["review_text"].str.split().str.len()
    r = r[r["n_words"] >= 2]
    log["reviews_with_text"] = int(orders["has_review_text"].sum())
    log["reviews_text_dedup_min2words"] = len(r)
    cols = [
        "review_id", "order_id", "customer_unique_id", "purchase_date", "order_month",
        "review_score", "review_text", "n_words", "category", "seller_id", "customer_state",
        "is_late", "delay_days",
    ]
    return r[cols].reset_index(drop=True)


def build_customers(orders: pd.DataFrame, log: dict) -> pd.DataFrame:
    """One row per customer, described by their FIRST order (the first experience),
    labelled by whether they ordered again within the repurchase window."""
    o = orders.sort_values("purchase_date")
    o["order_rank"] = o.groupby("customer_unique_id").cumcount()
    first = o[o["order_rank"] == 0].set_index("customer_unique_id")
    # A "return" is an order at least 1 day after the first one. ~30% of raw second orders
    # are same-day basket splits, which are not retention.
    o["first_date"] = o["customer_unique_id"].map(first["purchase_date"])
    later = o[(o["purchase_date"] - o["first_date"]) >= pd.Timedelta(days=1)]
    second = later.groupby("customer_unique_id")["purchase_date"].min()
    log["same_day_second_orders_ignored"] = int(
        ((o["order_rank"] == 1) & ((o["purchase_date"] - o["first_date"]) < pd.Timedelta(days=1))).sum()
    )

    c = first.copy()
    c["second_purchase_date"] = second.reindex(c.index)
    c["days_to_second"] = (c["second_purchase_date"] - c["purchase_date"]).dt.total_seconds() / 86400
    c["n_orders_total"] = o.groupby("customer_unique_id").size().reindex(c.index)

    # --- Point-in-time features: only what is known PREDICTION_DAY days after the first purchase.
    # The same cut-off applies to every customer, so missingness cannot leak the label.
    D = PREDICTION_DAY
    not_delivered = c["delivery_days"] > D
    overdue = not_delivered & (c["promised_days"] < D)
    c["delivered_by_t0"] = (~not_delivered).astype(int)
    c.loc[not_delivered, "delivery_days"] = np.nan
    c.loc[not_delivered, "is_late"] = overdue[not_delivered].astype(int)
    c.loc[not_delivered, "delay_days"] = np.where(
        overdue[not_delivered], D - c.loc[not_delivered, "promised_days"], np.nan
    )
    review_offset = (c["review_creation_date"] - c["purchase_date"]).dt.total_seconds() / 86400
    unseen = review_offset.isna() | (review_offset > D)
    c.loc[unseen, ["review_id", "review_score", "review_text"]] = np.nan
    c.loc[unseen, "has_review_text"] = 0

    # --- Label: returned in (D, D + window]. Customers who already returned before D are
    # "already retained" at prediction time and are excluded from the modelling population.
    data_end = orders["purchase_date"].max()
    c["returned_before_t0"] = (c["days_to_second"] <= D).astype(int)
    c["observable"] = (
        (c["purchase_date"] + pd.Timedelta(days=D + REPURCHASE_WINDOW_DAYS)) <= data_end
    ) & (c["returned_before_t0"] == 0)
    c["repurchased"] = (
        (c["days_to_second"] > D) & (c["days_to_second"] <= D + REPURCHASE_WINDOW_DAYS)
    ).astype(int)

    log["prediction_day"] = D
    log["customers_total"] = len(c)
    log["customers_observable"] = int(c["observable"].sum())
    log["repurchase_rate_observable"] = round(float(c.loc[c["observable"], "repurchased"].mean()), 4)
    log["positives_observable"] = int(c.loc[c["observable"], "repurchased"].sum())
    log["excluded_returned_before_t0"] = int(c["returned_before_t0"].sum())
    log["first_orders_not_delivered_by_t0"] = int(not_delivered.sum())
    log["first_reviews_not_visible_by_t0"] = int(unseen.sum())
    log["data_start"] = str(orders["purchase_date"].min().date())
    log["data_end"] = str(data_end.date())

    c = c.drop(columns=["order_rank"]).reset_index()
    return c


def main() -> None:
    log: dict = {}
    t = load_raw()
    orders = build_orders(t, log)
    reviews = build_reviews(orders, log)
    customers = build_customers(orders, log)

    orders.to_parquet(PROCESSED / "orders.parquet", index=False)
    reviews.to_parquet(PROCESSED / "reviews.parquet", index=False)
    customers.to_parquet(PROCESSED / "customers.parquet", index=False)

    (ARTIFACTS / "data_build_log.json").write_text(encoding="utf-8", data=json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
