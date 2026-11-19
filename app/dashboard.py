"""ChurnSight dashboard (aurora glass theme, plain language).  Run:  streamlit run app/dashboard.py"""

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
sys.path.insert(0, str(ROOT / "app"))

import theme as T  # noqa: E402
from churnsight.config import ARTIFACTS, PROCESSED  # noqa: E402
from churnsight.llm import client  # noqa: E402

st.set_page_config(page_title="ChurnSight", page_icon="✨", layout="wide")
T.apply()


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


eda = load_json("eda_summary.json") or {}
impact = load_csv("issue_impact.csv")
clusters = load_json("complaint_clusters.json") or {"clusters": []}
seg = load_csv("segment_profiles.csv")
hist = load_json("alerts_history.json") or []

PLAIN_TOPIC = {
    "delivery_delay": "Delivery took too long", "not_received": "Order never arrived",
    "defect_quality": "Broken or poor quality", "wrong_missing_item": "Wrong or missing item",
    "support_refund": "Refund, return or no reply", "fast_delivery": "Fast delivery",
    "good_quality": "Good quality", "praise_recommend": "Happy, would recommend",
}
COMPLAINTS = ["wrong_missing_item", "defect_quality", "support_refund", "delivery_delay", "not_received"]
RETURN_RATE = f"{eda.get('repurchase_rate_180d', 0) * 100:.1f} in 100"
N_TEXT_REVIEWS = eda.get("n_text_reviews_clean", 0)

T.brand()
tabs = st.tabs(["Overview", "Customer groups", "Who comes back", "What customers say",
                "Sellers and cost", "Ask the reviews", "Who to help first"])

# =================================================================== Overview
with tabs[0]:
    one_time = eda.get("share_one_time_customers", 0.97)
    T.hero(f"{round(one_time * 100)} out of 100 customers buy only once.",
           f"We read {N_TEXT_REVIEWS / 1000:.0f},000 customer reviews and {eda.get('n_orders', 0) / 1000:.0f},000 orders from a Brazilian online marketplace "
           "to find out why, and who to help first.")
    c = st.columns(4)
    with c[0]:
        T.card("Customers", f"{eda.get('n_customers', 0):,}", "bought between 2016 and 2018")
    with c[1]:
        T.card("Come back", RETURN_RATE, "order again within 7 months", T.RED)
    with c[2]:
        T.card("Parcels late", f"{eda.get('late_delivery_rate', 0) * 100:.0f}%", "arrive after the promised date", T.AMBER)
    with c[3]:
        T.card("Unhappy reviews", f"{clusters.get('n_complaints', 0):,}", "1 or 2 stars, with a comment")
    st.write("")

    left, right = st.columns(2)
    with left:
        on, late = eda.get("avg_review_score_on_time", 0), eda.get("avg_review_score_late", 0)
        st.subheader(f"A late parcel drops the rating from {on:.1f} to {late:.1f} stars")
        fig = go.Figure(go.Bar(x=["Arrived on time", "Arrived late"], y=[on, late], marker_color=[T.TEAL, T.RED],
                               text=[f"{on:.1f} ★", f"{late:.1f} ★"], textposition="outside", width=0.5))
        st.plotly_chart(T.style(fig).update_yaxes(range=[0, 5.5], title="Average stars"), width="stretch")
    with right:
        st.subheader("What unhappy customers complain about")
        agg = {}
        for cl in clusters.get("clusters", []):
            agg[cl.get("plain_name", "Other complaints")] = agg.get(cl.get("plain_name", "Other complaints"), 0) + cl["size"]
        if agg:
            s = pd.Series(agg).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=T.PURPLE,
                                   text=[f"{v:,}" for v in s.values], textposition="outside"))
            # Leave room on the right so the value labels are never cut off.
            st.plotly_chart(T.style(fig).update_xaxes(title="Unhappy reviews", range=[0, s.max() * 1.2]),
                            width="stretch")

    st.subheader("What we should do")
    a = st.columns(3)
    queue_preview = load_parquet("action_queue.parquet")
    n_high = int((queue_preview["priority"] == "high").sum()) if queue_preview is not None else 0
    hot = load_csv("seller_hotspots.csv", index_col=0)
    with a[0]:
        T.action(1, f"Call {n_high:,} customers first", "The most upset ones: their order never came, nobody answered, or they spent a lot.")
    with a[1]:
        worst = float(hot["complaint_rate_vs_platform"].max()) if hot is not None else 0
        T.action(2, f"Check {len(hot) if hot is not None else 0} sellers",
                 f"Their customers complain more than average, up to {worst:.1f} times as often.")
    with a[2]:
        T.action(3, "Plan for big sales", "On Black Friday 2017, 'my order never came' complaints nearly tripled.")

    with st.expander("Technical details"):
        coh = load_csv("cohort_retention.csv", index_col=0)
        if coh is not None:
            st.caption("Monthly retention by first-purchase cohort (% of the cohort ordering again)")
            z = coh.drop(columns="cohort_size")
            st.plotly_chart(T.style(px.imshow(z, color_continuous_scale="Purples", aspect="auto",
                                              labels=dict(x="Months since first purchase", y="Cohort", color="%"))), width="stretch")
        st.json(load_json("data_build_log.json") or {})

# =================================================================== Customer groups
NEXT_STEP = {
    "Recent Happy Buyers": "Say thank you and ask for a review while they still remember the order.",
    "Lapsed Happy Buyers": "They liked us but drifted away: a win-back email after about a year.",
    "Unhappy One-timers": "Apologise, fix the problem, then a small sorry voucher.",
    "Burned by Late Delivery": "Fix delivery first; a late-delivery voucher if it was more than a week late.",
    "Loyal Repeat Buyers": "Keep them close: early access and good service, no discounts needed.",
}
PLAIN_SEG = {
    "Recent Happy Buyers": "Happy, bought recently", "Lapsed Happy Buyers": "Happy, but long gone",
    "Unhappy One-timers": "Unhappy with the product", "Burned by Late Delivery": "Unhappy because it came late",
    "Loyal Repeat Buyers": "Loyal, buy again and again",
}
with tabs[1]:
    T.hero(f"Our customers fall into {len(seg) if seg is not None else 5} simple groups.",
           "Grouped by when they bought, how much they spent, how often, and how their first order went.")
    if seg is not None:
        seg = seg.sort_values("size", ascending=False)
        cols = st.columns(len(seg))
        for col, (_, r) in zip(cols, seg.iterrows()):
            colour = T.RED if r["avg_review_score"] < 3 else T.TEAL
            with col:
                T.card(PLAIN_SEG.get(r["segment"], r["segment"]), f"{r['share'] * 100:.0f}%",
                       f"{int(r['size']):,} people · {r['avg_review_score']:.1f}★ · R${r['monetary']:.0f} spent", colour)
        st.write("")
        st.subheader("Suggested next step for each group")
        for _, r in seg.iterrows():
            T.insight(f"{PLAIN_SEG.get(r['segment'], r['segment'])}: {NEXT_STEP.get(r['segment'], '')}")
        with st.expander("Technical details"):
            st.dataframe(seg.drop(columns=["segment_id"], errors="ignore").set_index("segment").round(2), width="stretch")
            m = load_json("segmentation_metrics.json")
            if m:
                st.caption(f"KMeans on RFM + experience features. Silhouette by k: "
                           f"{ {k: round(v, 3) for k, v in m['silhouette_by_k'].items()} }. k={m['chosen_k']} chosen: "
                           "k=3 scores higher but puts ~80% of customers in one group, which nobody can act on.")

# =================================================================== Who comes back
with tabs[2]:
    T.hero("Coming back is rare, and hard to predict.",
           f"Only {RETURN_RATE} customers order again. Our model still finds the likely ones faster than guessing.")
    gains = load_json("gains.json")
    left, right = st.columns([1.4, 1])
    with left:
        if gains:
            xs = [0] + [int(k) for k in gains]
            ys = [0] + [v * 100 for v in gains.values()]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="Our model", line=dict(color=T.PURPLE, width=3),
                                     fill="tozeroy", fillcolor="rgba(175,169,236,0.18)"))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random guess",
                                     line=dict(color=T.MUTED, dash="dash")))
            fig.update_xaxes(title="% of customers we contact (most likely first)")
            fig.update_yaxes(title="% of returning customers found")
            st.subheader("If we contact customers in our model's order…")
            st.plotly_chart(T.style(fig, 360), width="stretch")
    with right:
        st.write("")
        st.write("")
        if gains:
            T.card("Contact the top 20%", f"{gains['20'] * 100:.0f}%", "of all returning customers are found (random: 20%)", T.PURPLE)
        st.write("")
        T.card("After a 1–2 star review", "14% slower", "to come back than other customers", T.RED)
    T.insight("Honest finding: what people wrote in their review did not make the prediction better. "
              "The product they bought and the time of year matter more.")
    with st.expander("Technical details"):
        ab = load_csv("churn_ablation.csv")
        if ab is not None:
            st.dataframe(ab.set_index("variant"), width="stretch")
        imp = load_csv("shap_importance.csv", index_col=0)
        if imp is not None:
            top = imp.head(10).iloc[::-1]
            st.plotly_chart(T.style(go.Figure(go.Bar(x=top["mean_abs_shap"], y=top.index, orientation="h",
                                                     marker_color=T.PURPLE)), 320), width="stretch")
        cox = load_csv("survival_cox.csv", index_col=0)
        if cox is not None:
            st.caption("Cox proportional hazards (hazard ratio < 1 = slower to return)")
            st.dataframe(cox, width="stretch")
        km = ARTIFACTS / "figures" / "km_by_experience.png"
        if km.exists():
            st.image(str(km), caption="Kaplan-Meier: time to second purchase", width="stretch")

# =================================================================== What customers say
with tabs[3]:
    T.hero("We read every review automatically.",
           "A language model trained on Portuguese reviews sorts each one into topics, and warns us when a problem suddenly grows.")
    nlp_cmp = load_csv("nlp_comparison.csv")
    f1_model = float(nlp_cmp["topic_macro_f1"].iloc[-1]) if nlp_cmp is not None else 0
    f1_rules = float(nlp_cmp["topic_macro_f1"].iloc[0]) if nlp_cmp is not None else 0
    c = st.columns(3)
    with c[0]:
        T.card("Reviews read", f"{N_TEXT_REVIEWS:,}", "in Portuguese, 2016–2018")
    with c[1]:
        T.card("Topic score", f"{f1_model * 100:.0f} / 100",
               f"F1 on 250 held-out reviews; simple keyword rules: {f1_rules * 100:.0f}", T.TEAL)
    with c[2]:
        T.card("Problem warnings", f"{len(hist)}", "would have been sent, month by month", T.AMBER)
    st.write("")

    trends = load_csv("topic_trends.csv", index_col=0)
    if trends is not None:
        st.subheader("How often each problem was mentioned, month by month")
        pick = st.radio("Problem", COMPLAINTS[::-1], horizontal=True, format_func=lambda t: PLAIN_TOPIC[t])
        alert_months = {h["month"] for h in hist if h["topic"] == pick}
        ser = trends[pick] * 100
        fig = go.Figure(go.Bar(x=ser.index, y=ser.values,
                               marker_color=[T.AMBER if m in alert_months else "rgba(175,169,236,0.55)" for m in ser.index],
                               hovertemplate="%{x}: %{y:.1f}% of reviews<extra></extra>"))
        st.plotly_chart(T.style(fig, 300).update_yaxes(title="% of reviews"), width="stretch")
        st.caption("Yellow bars = months when ChurnSight would have warned the team.")

    st.subheader("Warnings it would have sent")
    big = sorted(hist, key=lambda h: -h["z_score"])[:4]
    w = st.columns(len(big) or 1)
    for col, h in zip(w, big):
        with col:
            times = h["share"] / h["baseline_share"] if h["baseline_share"] else 0
            T.card(pd.Period(h["month"]).strftime("%b %Y"), f"{times:.1f}× more", PLAIN_TOPIC.get(h["topic"], h["label"]), T.AMBER)
    st.write("")
    T.insight("These warnings match reality: in November 2017 (Black Friday) orders jumped 63% and late parcels went "
              "from 5% to 14%. The review system never saw delivery data, yet it flagged exactly those months.")

    st.subheader("Biggest complaint groups")
    shown = 0
    for cl in sorted(clusters.get("clusters", []), key=lambda c: -c["size"]):
        if shown == 8:
            break
        shown += 1
        with st.expander(f"{cl.get('plain_name', 'Complaints')} · {cl['size']} reviews"):
            for q in cl["representative_reviews"][:3]:
                T.quote(q)
            st.caption("Reviews are in Portuguese. Key words: " + ", ".join(cl["keywords"][:6]))

    with st.expander("Technical details"):
        if nlp_cmp is not None:
            st.dataframe(nlp_cmp.set_index("model"), width="stretch")
        st.dataframe(pd.DataFrame(hist), width="stretch", hide_index=True)

# =================================================================== Sellers and cost
with tabs[4]:
    T.hero("A few sellers cause a lot of the pain.",
           "For some sellers, most of their reviews are complaints. Customers who complained also came back "
           "a little less often, though that gap is small.")
    if hot is not None:
        top = hot.head(10).copy()
        top["name"] = [f"Seller {chr(65 + i)} · {cat.replace('_', ' ')}" for i, cat in enumerate(top["main_category"].fillna("various").replace("unknown", "various"))]
        avg = float(top["complaint_rate"].iloc[0] / top["complaint_rate_vs_platform"].iloc[0])
        fig = go.Figure(go.Bar(x=top["complaint_rate"] * 100, y=top["name"], orientation="h", marker_color=T.RED,
                               text=[f"{v * 10:.0f} in 10" for v in top["complaint_rate"]], textposition="outside"))
        fig.add_vline(x=avg * 100, line_dash="dash", line_color=T.MUTED, annotation_text="average seller",
                      annotation_position="bottom right", annotation_font_color=T.AMBER)
        st.subheader("Reviews with a complaint, by seller")
        st.plotly_chart(T.style(fig, 400).update_yaxes(autorange="reversed")
                        .update_xaxes(title="% of reviews with a complaint", range=[0, top["complaint_rate"].max() * 120]),
                        width="stretch")
    if impact is not None:
        st.subheader("Does a bad first experience stop people coming back?")
        rows = impact[impact["topic"].isin(COMPLAINTS)]
        c = st.columns(len(rows))
        for col, (_, r) in zip(c, rows.iterrows()):
            with col:
                T.card(PLAIN_TOPIC[r["topic"]], f"{r['repurchase_rate'] * 1000:.0f} in 1,000",
                       f"came back (others: {r['repurchase_rate_others'] * 1000:.0f})",
                       T.RED if r["repurchase_rate"] < r["repurchase_rate_others"] else T.TEAL)
        st.write("")
        T.insight("Broken and wrong items hurt the most, but the differences are small and not certain. "
                  "In this marketplace almost everyone buys once anyway, so the biggest win is fixing problems fast.")
        with st.expander("Technical details"):
            st.dataframe(impact.set_index("label"), width="stretch")
            if hot is not None:
                st.dataframe(hot, width="stretch")

# =================================================================== Ask the reviews
with tabs[5]:
    from churnsight.rag.retriever import dense_available

    T.hero("Ask the reviews anything.", "Type a question in English. We find the reviews that talk about it, even though they are in Portuguese.")
    if not dense_available():
        st.warning("Light version: search with Portuguese words, e.g. 'faltando', 'não recebi', 'quebrado'. "
                   "The full version understands English questions.")
    examples = ["I ordered several items but only part of the order arrived", "The product arrived broken",
                "I asked for a refund and nobody answered", "The delivery took more than a month"]
    q = st.selectbox("Try an example, or pick one and edit it below", examples)
    q = st.text_input("Your question", q)
    cols = st.columns(3)
    stars = cols[0].slider("Star rating of the review", 1, 5, (1, 3))
    topic = cols[1].selectbox("Only this problem", ["any", *COMPLAINTS], format_func=lambda t: "Any problem" if t == "any" else PLAIN_TOPIC[t])
    months = cols[2].text_input("From month (YYYY-MM)", "2017-01")
    if st.button("Search reviews", type="primary"):
        from churnsight.nlp.weak_labels import topic_thresholds
        from churnsight.rag.answer import ask

        filters = {"review_score": {"min": stars[0], "max": stars[1]}, "order_month": {"min": months}}
        if topic != "any":
            filters[f"topic_{topic}"] = {"min": topic_thresholds()[topic]}
        with st.spinner("Reading reviews..."):
            res = ask(q, filters=filters)
        if res.llm_used:
            st.markdown(res.answer)
        st.subheader(f"{len(res.reviews)} matching reviews")
        for r in res.reviews.itertuples():
            T.quote(f"{r.review_text}  ·  {int(r.review_score)}★ · {r.order_month}")
    with st.expander("Technical details"):
        ev = load_csv("retrieval_eval.csv", index_col=0)
        if ev is not None:
            st.dataframe(ev, width="stretch")
            st.caption("Hybrid search (BM25 + multilingual embeddings) on 16 queries; relevance comes from the 250 gold reviews' topic labels.")

# =================================================================== Who to help first
with tabs[6]:
    T.hero("Who to help first.", "Every unhappy customer gets a priority and, if the rules allow, a sorry voucher.")
    queue = load_parquet("action_queue.parquet")
    if queue is None:
        st.info("Run `python -m churnsight.llm.queue` to build the list.")
    else:
        c = st.columns(3)
        counts = queue["priority"].value_counts()
        with c[0]:
            T.card("Call today", f"{counts.get('high', 0):,}", "very upset or big orders", T.RED)
        with c[1]:
            T.card("Email this week", f"{counts.get('medium', 0):,}", "unhappy, less urgent", T.AMBER)
        with c[2]:
            T.card("Keep an eye on", f"{counts.get('low', 0):,}", "mixed feelings", T.TEAL)
        st.write("")
        lvl = st.radio("Show", ["high", "medium"], horizontal=True,
                       format_func=lambda v: "Call today" if v == "high" else "Email this week")
        sub = queue[queue["priority"] == lvl].head(200).reset_index(drop=True)
        labels = [f"{r.review_text[:70]}…  ·  R${r.payment_value:.0f}" for r in sub.itertuples()]
        i = st.selectbox("Pick a customer", range(len(sub)), format_func=lambda k: labels[k])
        row = sub.iloc[i]
        from churnsight.llm.actions import Case, suggest

        case = Case(review_id=row["review_id"], review_text=row["review_text"], review_score=row["review_score"],
                    topics=list(row["topics"]), payment_value=row["payment_value"],
                    delay_days=None if pd.isna(row["delay_days"]) else row["delay_days"])
        s = (load_json("action_examples.json") or {}).get(row["review_id"]) or suggest(case, use_llm=False).to_dict()

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("What happened")
            T.quote(row["review_text"])
            probs = ", ".join(PLAIN_TOPIC[t] for t in row["topics"] if t in PLAIN_TOPIC) or "general complaint"
            late = "" if pd.isna(row["delay_days"]) or row["delay_days"] <= 0 else f" · {row['delay_days']:.0f} days late"
            st.markdown(f"**{int(row['review_score'])}★** · order of **R${row['payment_value']:.0f}**{late}  \nProblem: **{probs}**")
        with right:
            st.subheader("What we should do")
            T.card("Priority", "Call today" if s["priority"] == "high" else "Email this week" if s["priority"] == "medium" else "Monitor",
                   "; ".join(s["priority_reasons"]), T.RED if s["priority"] == "high" else T.AMBER)
            st.write("")
            T.card("Sorry voucher", f"R${s['eligible_voucher_brl']}", s["voucher_reason"], T.TEAL)
        if s.get("reply_pt"):
            st.subheader("Suggested reply")
            T.quote(s["reply_pt"])
            st.caption(s.get("reply_en", ""))
        elif not client.available():
            st.caption("Writing the reply automatically needs an AI key. Priority and voucher come from fixed, auditable rules.")
        with st.expander("Which company rules were used"):
            for ch in s["policy_chunks"]:
                st.markdown(f"**{ch['chunk_id']}**\n\n{ch['text']}")
