# ChurnSight

**Customer 360 intelligence: link what customers *say* (reviews) with what they *do* (orders) to explain churn and decide what to do next.**

Built on the public [Olist Brazilian E-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (~100k orders, 2016–2018, reviews in Portuguese).

<!-- RESULTS -->

## Why this project

E-commerce teams have two disconnected views of customers:

| | What customers **say** | What customers **do** |
|---|---|---|
| Data | Reviews, complaints (unstructured text) | Orders, payments, deliveries (tables) |
| Usual project | Sentiment analysis | RFM segmentation / churn model |
| Question answered | "What are people unhappy about?" | "Who is likely to leave?" |

ChurnSight joins them per customer, so it can answer the question a business actually cares about: **which problems cost us returning customers, how much, and what should we do about each unhappy customer?**

## What it does

| Module | What | Techniques |
|---|---|---|
| 1. Review intelligence | Tag every review with 8 topics + negativity; discover recurring complaint clusters | Weak supervision (Portuguese keyword rules), fine-tuned **BERTimbau** (multi-label), multilingual sentence embeddings, **BERTopic** |
| 2. Segmentation | 5 actionable customer segments | RFM + experience features, KMeans, silhouette vs usefulness trade-off |
| 3. Retention model | Who comes back after the first order, and why | Point-in-time features, LightGBM, **SHAP**, Kaplan-Meier + **Cox** survival |
| 4. Issue impact | Repurchase gap and revenue at stake per complaint topic; seller hotspots; rising-issue alerts | Bootstrap CIs, z-score spike detection |
| 5. Ask ChurnSight (RAG) | Ask questions in English about Portuguese reviews, answered with verified citations | Hybrid retrieval (BM25 + dense, RRF), cross-encoder rerank, metadata filters, **LLM** |
| 6. Action Center | Prioritised queue of unhappy customers with a policy-grounded reply draft | Rule engine for priority/vouchers, policy RAG, LLM structured output, guardrails |

Delivered as a **Streamlit dashboard** and a **FastAPI** service.

## Architecture

```
Olist CSVs (9 tables)
   │  data/build.py: join, clean, point-in-time features (day 30), label = return in days 30-210
   ▼
orders / customers / reviews (parquet)
   ├── analytics/cohorts.py ── cohort retention, EDA
   ├── analytics/segments.py ─ RFM + experience → KMeans → named segments
   ├── nlp/weak_labels.py ──── Portuguese keyword rules → noisy topic labels
   │      └── nlp/train_classifier.py ─ BERTimbau fine-tune → topic + negativity probabilities
   ├── nlp/clusters.py ─────── multilingual embeddings → BERTopic on 1-2★ reviews
   │      └── llm/insights.py ─ LLM names clusters, quotes verified against reviews
   ├── analytics/churn.py ──── LightGBM (A: orders · B: +score · C: +review NLP), SHAP, Cox
   ├── analytics/impact.py ─── topic → repurchase gap → revenue at stake; trends + alerts
   ├── rag/ ────────────────── hybrid retriever, retrieval eval, Ask ChurnSight
   └── llm/actions.py ──────── priority + voucher rules → policy RAG → LLM reply → guardrails
                                   │
                     app/dashboard.py (Streamlit) · app/api.py (FastAPI)
```

## Key design decisions

- **Point-in-time features (no leakage).** Every customer is scored 30 days after their first purchase, using only what was known by then (delivery status, a review written by day 30). The label is "returned between day 30 and day 210". An earlier version masked reviews written *after* a second purchase; that masking only ever hit returning customers, so "review missing" leaked the label. The fixed cut-off applies to everyone equally.
- **Same-day orders are not retention.** About 30% of raw "second orders" were placed the same day as the first (basket splits). A return needs to be at least 1 day later.
- **Weak supervision instead of hand-labelling 30k reviews.** Portuguese keyword rules label the training set; a fine-tuned transformer learns to generalise beyond the keywords. Everything is evaluated on a separate, hand-labelled gold set of 250 reviews (50 per star rating) that is never used for training.
- **Usefulness over a single metric.** Silhouette prefers k=3 segments, but that puts ~80% of customers in one group. We pick the best k ≥ 5 and say so.
- **Rules decide money, the LLM writes words.** In the Action Center the voucher amount comes from a rule engine that mirrors the goodwill policy. The LLM only drafts the reply. Guardrails reject drafts that change the amount, cite policies that were not retrieved, mention unapproved R$ amounts, or skip escalation on legal/safety language.
- **Exact search over an ANN index.** ~34k × 384-dim vectors: brute-force cosine takes milliseconds, so a vector database would add moving parts without benefit at this scale.
- **Works without an API key.** With `LLM_PROVIDER=none` every LLM feature falls back to retrieval-only or cached outputs, so the dashboard, API and tests run anywhere.

## Run it

```bash
# 1. Environment (Python 3.11+). GPU optional but recommended for the classifier.
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or the CPU wheel
pip install -r requirements-dev.txt && pip install -e .

# 2. Full pipeline (downloads the public dataset on first run)
python -m churnsight.pipeline

# 3. Optional: LLM-powered features
copy .env.example .env      # set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY
python -m churnsight.llm.insights          # name complaint clusters
python -m churnsight.llm.queue --draft 10  # pre-draft demo replies

# 4. Apps
streamlit run app/dashboard.py
uvicorn app.api:app --reload               # http://localhost:8000/docs

# 5. Tests
pytest -q
```

The deployed dashboard runs in **lite mode** (`CHURNSIGHT_LITE=1`): it reads the slim bundle in `data/app/` and uses BM25 search instead of the neural encoder, so it fits a 512 MB container. See `Dockerfile` and `render.yaml`.

## Project layout

```
src/churnsight/
  data/build.py            raw tables → clean, point-in-time customer table
  analytics/               cohorts, segments, churn model + survival, issue impact + alerts
  nlp/                     weak labels, BERTimbau classifier, embeddings + BERTopic
  rag/                     hybrid retriever, policy KB, Ask ChurnSight, retrieval eval
  llm/                     LLM client, cluster insights, Action Center, outreach queue
app/                       Streamlit dashboard, FastAPI
docs/policies/             synthetic policy knowledge base for a fictional marketplace
data/labels/               gold annotations (hand-labelled evaluation set)
artifacts/                 metrics, tables and figures produced by the pipeline
tests/                     unit tests for rules, guardrails, spike detection
```

## Data and licence

- Olist Brazilian E-commerce Public Dataset by Olist, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The slim bundle in `data/app/` is derived from it and shared under the same licence, for non-commercial use.
- `docs/policies/` describes **Aurora Marketplace, a fictional company** invented for this demo. It is not Olist's policy.
- Gold labels in `data/labels/gold_annotations.txt` were produced with an LLM reading each review, single pass; treat them as a careful but unaudited reference.
