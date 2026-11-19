"""ChurnSight dashboard.  Run:  streamlit run app/dashboard.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnsight.config import ARTIFACTS, PROCESSED  # noqa: E402
from churnsight.llm import client  # noqa: E402
from churnsight.nlp.weak_labels import TOPICS  # noqa: E402

st.set_page_config(page_title="ChurnSight", page_icon="📉", layout="wide")


# ---------- data loading ----------
@st.cache_data
def load_json(name: str):
    p = ARTIFACTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_csv(name: str, **kw):
    p = ARTIFACTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def load_parquet(name: str):
    p = PROCESSED / name
    return pd.read_parquet(p) if p.exists() else None


def fig_or_note(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Run the pipeline to generate: {path.name}")


eda = load_json("eda_summary.json") or {}
build = load_json("data_build_log.json") or {}

st.title("ChurnSight")
st.caption(
    "What customers **say** (reviews) + what they **do** (orders) → who stays, why they leave, "
    "and what to do next. Data: Olist Brazilian E-commerce (2016-2018)."
)

tabs = st.tabs(["Overview", "Segments", "Retention model", "Review intelligence", "Issue impact",
                "Ask ChurnSight", "Action Center"])

# ---------- Overview ----------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Delivered orders", f"{eda.get('n_orders', 0):,}")
    c[1].metric("Customers", f"{eda.get('n_customers', 0):,}")
    c[2].metric("One-time buyers", f"{eda.get('share_one_time_customers', 0):.0%}")
    c[3].metric("Return within 30-210 days", f"{eda.get('repurchase_rate_180d', 0):.2%}")
    c[4].metric("Late deliveries", f"{eda.get('late_delivery_rate', 0):.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Late delivery wrecks the review score")
        st.plotly_chart(
            go.Figure(go.Bar(x=["On time", "Late"],
                             y=[eda.get("avg_review_score_on_time", 0), eda.get("avg_review_score_late", 0)],
                             marker_color=["#4C9A6A", "#C44E52"], text=[eda.get("avg_review_score_on_time"), eda.get("avg_review_score_late")]))
            .update_layout(yaxis_title="Average review score", yaxis_range=[0, 5], height=320),
            width="stretch",
        )
    with right:
        st.subheader("Repurchase rate by first review score")
        by = eda.get("repurchase_rate_by_review_score", {})
        st.plotly_chart(
            px.bar(x=list(by.keys()), y=[v * 100 for v in by.values()], labels={"x": "Stars", "y": "% returning"})
            .update_layout(height=320), width="stretch",
        )
    st.subheader("Cohort retention")
    fig_or_note(ARTIFACTS / "figures" / "cohort_retention.png", "% of each first-purchase cohort ordering again, by month")
    with st.expander("Data pipeline log"):
        st.json(build)

# ---------- Segments ----------
with tabs[1]:
    prof = load_csv("segment_profiles.csv")
    if prof is not None:
        st.subheader("Customer segments (RFM + experience, KMeans)")
        st.plotly_chart(px.bar(prof.sort_values("size"), x="size", y="segment", orientation="h", text="share")
                        .update_layout(height=320), width="stretch")
        st.dataframe(prof.drop(columns=["segment_id"], errors="ignore").set_index("segment").round(2), width="stretch")
        seg_m = load_json("segmentation_metrics.json")
        if seg_m:
            st.caption(f"Silhouette by k: { {k: round(v, 3) for k, v in seg_m['silhouette_by_k'].items()} }. "
                       f"k={seg_m['chosen_k']} chosen: k=3 scores higher but puts ~80% of customers in one segment.")

# ---------- Retention model ----------
with tabs[2]:
    ab = load_csv("churn_ablation.csv")
    cm = load_json("churn_metrics.json")
    if ab is not None:
        st.subheader("Does what customers SAY improve the retention model?")
        st.dataframe(ab.set_index("variant"), width="stretch")
        if cm:
            st.caption(f"Time-based split at {cm['time_split_cutoff']} (train {cm['n_train']:,} / test {cm['n_test']:,}). "
                       "Features use only information known 30 days after the first purchase.")
    left, right = st.columns(2)
    with left:
        imp = load_csv("shap_importance.csv", index_col=0)
        if imp is not None:
            st.subheader("What drives returning (SHAP)")
            top = imp.head(12).iloc[::-1]
            st.plotly_chart(px.bar(top, x="mean_abs_shap", y=top.index, orientation="h").update_layout(height=420, yaxis_title=None),
                            width="stretch")
    with right:
        st.subheader("Time to second purchase")
        fig_or_note(ARTIFACTS / "figures" / "km_by_experience.png", "Kaplan-Meier, by first experience")
    cox = load_csv("survival_cox.csv", index_col=0)
    if cox is not None:
        st.caption("Cox proportional hazards (hazard ratio < 1 = slower to return)")
        st.dataframe(cox, width="stretch")

# ---------- Review intelligence ----------
with tabs[3]:
    nlp = load_csv("nlp_comparison.csv")
    if nlp is not None:
        st.subheader("Topic classifier: keyword rules vs TF-IDF vs fine-tuned BERTimbau (hand-labelled gold set)")
        st.dataframe(nlp.set_index("model"), width="stretch")
    trends = load_csv("topic_trends.csv", index_col=0)
    if trends is not None:
        st.subheader("Share of reviews mentioning each topic, by month")
        pick = st.multiselect("Topics", list(TOPICS), default=["delivery_delay", "not_received", "defect_quality", "wrong_missing_item"],
                              format_func=lambda t: TOPICS[t])
        if pick:
            st.plotly_chart(px.line(trends[pick] * 100, labels={"value": "% of reviews", "order_month": "Month"})
                            .update_layout(height=380), width="stretch")
    alerts = load_json("alerts.json")
    if alerts is not None:
        st.subheader("Rising-issue alerts (latest month, z-score ≥ 2)")
        if alerts:
            for a in alerts:
                st.error(f"**{a['label']}**: {a['share']:.1%} of reviews in {a['month']} vs {a['baseline_share']:.1%} baseline "
                         f"(z = {a['z_score']}, {a['change_pct']:+.0f}%)")
        else:
            st.success("No complaint topic is spiking in the latest month (2018-08).")
    hist = load_json("alerts_history.json")
    if hist:
        with st.expander(f"Backtest: alerts the monthly report would have raised ({len(hist)})", expanded=True):
            st.dataframe(pd.DataFrame(hist)[["month", "label", "share", "baseline_share", "z_score", "change_pct"]],
                         width="stretch", hide_index=True)
            st.caption("Validated against order data the NLP never saw: in Nov 2017 (Black Friday) orders jumped 63% "
                       "and the late-delivery rate went from 5% to 14%; in Feb-Mar 2018 it reached 16-21%. "
                       "The review-based alerts fire in exactly those months.")
    ins = load_json("cluster_insights.json")
    raw = load_json("complaint_clusters.json")
    st.subheader("Recurring complaint clusters (BERTopic on 1-2 star reviews)")
    if ins:
        for c in sorted(ins["clusters"], key=lambda c: -c["size"]):
            with st.expander(f"{c['name']} · {c['size']} complaints · {c['severity']} · {c['owner_team']}"):
                st.write(c["summary"])
                st.markdown(f"**Recommended action:** {c['recommended_action']}")
                for q in c["evidence_quotes_pt"]:
                    st.markdown(f"> {q}")
                st.caption(f"Keywords: {', '.join(c['keywords'][:8])}")
    elif raw:
        st.caption("LLM naming not generated yet; showing BERTopic keywords.")
        for c in sorted(raw["clusters"], key=lambda c: -c["size"]):
            with st.expander(f"Cluster {c['cluster']} · {c['size']} complaints · {', '.join(c['keywords'][:4])}"):
                for t in c["representative_reviews"]:
                    st.markdown(f"> {t}")

# ---------- Issue impact ----------
with tabs[4]:
    imp = load_csv("issue_impact.csv")
    if imp is not None:
        st.subheader("Which first-order experiences change the chance of coming back?")
        fig = go.Figure(go.Bar(
            x=imp["diff_pp"], y=imp["label"], orientation="h",
            error_x=dict(type="data", symmetric=False, array=imp["ci95_high_pp"] - imp["diff_pp"],
                         arrayminus=imp["diff_pp"] - imp["ci95_low_pp"]),
            marker_color=["#C44E52" if d < 0 else "#4C9A6A" for d in imp["diff_pp"]],
        ))
        fig.update_layout(xaxis_title="Repurchase rate vs other reviewers (percentage points, 95% CI)", height=420)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(imp.set_index("label"), width="stretch")
    hot = load_csv("seller_hotspots.csv", index_col=0)
    if hot is not None:
        st.subheader("Seller hotspots (≥ 50 text reviews)")
        st.dataframe(hot, width="stretch")

# ---------- Ask ChurnSight ----------
with tabs[5]:
    from churnsight.rag.retriever import dense_available

    st.subheader("Ask a question about what customers say")
    if not dense_available():
        st.warning("Lite mode: keyword (BM25) search only. Reviews are in Portuguese, so search with Portuguese "
                   "words (e.g. *faltando item*, *não recebi*, *quebrado*). The full hybrid search runs locally.")
    q = st.text_input("Question", "I ordered several items but only part of the order arrived")
    cols = st.columns(4)
    stars = cols[0].slider("Review stars", 1, 5, (1, 3))
    months = cols[1].text_input("From month (YYYY-MM)", "2017-01")
    topic = cols[2].selectbox("Topic filter (classifier)", ["any", *TOPICS],
                              format_func=lambda t: "Any topic" if t == "any" else TOPICS[t])
    rerank = cols[3].toggle("Cross-encoder rerank", value=False, help="Did not improve nDCG in our evaluation")
    ev = load_csv("retrieval_eval.csv", index_col=0)
    if ev is not None:
        with st.expander("How good is retrieval? (16 queries, exact relevance from the hand-labelled gold set)"):
            st.dataframe(ev, width="stretch")
            st.caption("BM25 cannot match English questions to Portuguese reviews (nDCG 0.09); multilingual "
                       "embeddings can. Hybrid fusion is best overall; the cross-encoder did not help.")
    if st.button("Search", type="primary"):
        from churnsight.rag.answer import ask

        from churnsight.nlp.weak_labels import topic_thresholds

        filters = {"review_score": {"min": stars[0], "max": stars[1]}, "order_month": {"min": months}}
        if topic != "any":  # metadata filter on the fine-tuned classifier's prediction
            filters[f"topic_{topic}"] = {"min": topic_thresholds()[topic]}
        with st.spinner("Retrieving..."):
            res = ask(q, filters=filters, rerank=rerank)
        if res.llm_used:
            st.markdown(res.answer)
            st.caption(f"Confidence: {res.confidence} · cited {len(res.cited_review_ids)} reviews"
                       + (f" · dropped invalid citations: {res.invalid_citations}" if res.invalid_citations else ""))
        else:
            st.info("LLM answer disabled (no API key configured). Showing the retrieved evidence.")
        show = ["review_text", "review_score", "order_month", "category", "score", "review_id"]
        st.dataframe(res.reviews[show], width="stretch", hide_index=True,
                     column_config={"review_text": st.column_config.TextColumn("Review (pt-BR)", width="large")})

# ---------- Action Center ----------
with tabs[6]:
    st.subheader("Prioritised outreach queue")
    queue = load_parquet("action_queue.parquet")
    if queue is None:
        st.info("Run `python -m churnsight.llm.queue` to build the queue.")
    else:
        lvl = st.radio("Priority", ["high", "medium"], horizontal=True)
        sub = queue[queue["priority"] == lvl].head(200)
        st.dataframe(sub[["review_id", "review_score", "payment_value", "topics_label", "priority_reasons", "review_text"]],
                     width="stretch", hide_index=True)
        rid = st.selectbox("Open case", sub["review_id"])
        row = sub.set_index("review_id").loc[rid]
        st.markdown(f"> {row['review_text']}")
        from churnsight.llm.actions import Case, suggest

        case = Case(review_id=rid, review_text=row["review_text"], review_score=row["review_score"],
                    topics=list(row["topics"]), payment_value=row["payment_value"],
                    delay_days=None if pd.isna(row["delay_days"]) else row["delay_days"])
        cached = load_json("action_examples.json") or {}
        if rid in cached:
            s = cached[rid]
        elif st.button("Draft reply", type="primary", disabled=not client.available()):
            with st.spinner("Drafting..."):
                s = suggest(case).to_dict()
        else:
            s = suggest(case, use_llm=False).to_dict()
        c1, c2 = st.columns(2)
        c1.metric("Priority", s["priority"].upper())
        c2.metric("Eligible voucher (rules)", f"R${s['eligible_voucher_brl']}")
        st.caption(" · ".join(s["priority_reasons"]) + f" · {s['voucher_reason']}")
        if s.get("reply_pt"):
            left, right = st.columns(2)
            left.markdown("**Reply (pt-BR)**")
            left.write(s["reply_pt"])
            right.markdown("**Translation**")
            right.write(s["reply_en"])
            st.markdown(f"**Next step:** {s['resolution']}")
            if s["escalate_to_human"]:
                st.warning(f"Escalate to a human: {s['escalation_reason']}")
            if s["guardrail_flags"]:
                st.error("Guardrail flags: " + "; ".join(s["guardrail_flags"]))
        elif not client.available():
            st.info("Reply drafting needs an LLM key (LLM_PROVIDER=anthropic). Priority, voucher and policies above are rule-based.")
        with st.expander("Policy sections retrieved"):
            for ch in s["policy_chunks"]:
                st.markdown(f"**{ch['chunk_id']}**\n\n{ch['text']}")
