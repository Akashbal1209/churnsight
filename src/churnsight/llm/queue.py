"""Build the Action Center queue: first-order customers with a negative review, prioritised by rules.
Optionally pre-draft replies for a handful of demo cases (requires an LLM key).

Outputs: data/processed/action_queue.parquet, artifacts/action_examples.json (when LLM enabled)
Run:  python -m churnsight.llm.queue [--draft N]
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from churnsight.config import ARTIFACTS, PROCESSED
from churnsight.llm import client
from churnsight.llm.actions import Case, prioritise, suggest
from churnsight.nlp.weak_labels import TOPICS, topic_thresholds




def build_queue() -> pd.DataFrame:
    customers = pd.read_parquet(PROCESSED / "customers.parquet")
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")[["review_id", "review_text", "order_month", "category"]]
    nlp = pd.read_parquet(PROCESSED / "review_nlp.parquet")
    c = customers[customers["review_id"].notna()][
        ["customer_unique_id", "review_id", "review_score", "payment_value", "delay_days"]
    ].merge(reviews, on="review_id").merge(nlp, on="review_id")
    th = topic_thresholds()
    c = c[(c["review_score"] <= 2) | (c["sent_neg"] >= th["negative"])].copy()
    topic_cols = [f"topic_{t}" for t in TOPICS]
    c["topics"] = c[topic_cols].apply(lambda r: [t for t in TOPICS if r[f"topic_{t}"] >= th[t]], axis=1)
    c["topics_label"] = c["topics"].map(lambda ts: ", ".join(TOPICS[t] for t in ts))
    pr = c.apply(lambda r: prioritise(_case(r)), axis=1)
    c["priority"] = pr.map(lambda x: x[0])
    c["priority_reasons"] = pr.map(lambda x: "; ".join(x[1]))
    order = {"high": 0, "medium": 1, "low": 2}
    c = c.sort_values(["priority", "payment_value"], key=lambda s: s.map(order) if s.name == "priority" else -s)
    keep = ["customer_unique_id", "review_id", "review_score", "payment_value", "delay_days", "order_month",
            "category", "review_text", "topics", "topics_label", "priority", "priority_reasons", "sent_neg"]
    return c[keep].reset_index(drop=True)


def _case(r) -> Case:
    return Case(
        review_id=r["review_id"], review_text=r["review_text"], review_score=float(r["review_score"]),
        topics=list(r["topics"]), payment_value=float(r["payment_value"]),
        delay_days=None if pd.isna(r["delay_days"]) else float(r["delay_days"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=int, default=0, help="pre-draft replies for N demo cases (needs LLM)")
    args = ap.parse_args()
    q = build_queue()
    q.to_parquet(PROCESSED / "action_queue.parquet", index=False)
    print(q["priority"].value_counts().to_string())

    if args.draft and client.available():
        # A varied demo set: top cases per main topic.
        picks = []
        for t in ["not_received", "defect_quality", "wrong_missing_item", "delivery_delay", "support_refund"]:
            picks += q[q["topics"].map(lambda ts, t=t: t in ts)].head(max(1, args.draft // 5))["review_id"].tolist()
        examples, flagged = {}, 0
        for rid in dict.fromkeys(picks):
            row = q.set_index("review_id").loc[rid].copy()
            row["review_id"] = rid
            s = suggest(_case(row)).to_dict()
            flagged += bool(s["guardrail_flags"])
            examples[rid] = s
            print(rid, s["priority"], f"R${s['eligible_voucher_brl']}", "flags:", s["guardrail_flags"])
        (ARTIFACTS / "action_examples.json").write_text(encoding="utf-8", data=json.dumps(examples, indent=2, ensure_ascii=False))
        print(f"{len(examples)} drafts, {flagged} with guardrail flags")


if __name__ == "__main__":
    main()
