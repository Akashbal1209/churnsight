"""Repurchase (retention) model + explanations + survival analysis.

Question: given a customer's FIRST order experience, will they buy again within the window?
(Olist is ~97% one-time buyers, so "who comes back" is the meaningful retention target.)

Ablation (the key result): does adding what customers SAID (review NLP features)
improve prediction over what they DID (order/delivery features)?

Outputs (artifacts/):
  churn_metrics.json, churn_ablation.csv, shap_importance.csv,
  figures/shap_summary.png, figures/km_by_experience.png, survival_cox.csv
  data/processed/customer_scores.parquet (per-customer repurchase probability)

Run:  python -m churnsight.analytics.churn
"""

from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score

from churnsight.config import ARTIFACTS, PROCESSED, RANDOM_SEED

FIG = ARTIFACTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ORDER_FEATURES = [
    "items_price", "freight", "freight_ratio", "payment_value", "installments", "n_items",
    "n_sellers", "same_state_seller", "delivery_days", "promised_days", "delay_days", "is_late",
    "purchase_month", "payment_type", "category", "customer_state",
]
REVIEW_SCORE_FEATURES = ["review_score", "has_review_text"]
CATEGORICAL = ["payment_type", "category", "customer_state"]


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    c = pd.read_parquet(PROCESSED / "customers.parquet")
    c = c[c["observable"]].copy()
    c["purchase_month"] = c["purchase_date"].dt.month
    top_cats = c["category"].value_counts().index[:25]
    c["category"] = c["category"].where(c["category"].isin(top_cats), "other")
    for col in CATEGORICAL:
        c[col] = c[col].fillna("unknown").astype("category")

    nlp_cols: list[str] = []
    nlp_path = PROCESSED / "review_nlp.parquet"
    if nlp_path.exists():
        nlp = pd.read_parquet(nlp_path)
        nlp_cols = [col for col in nlp.columns if col.startswith(("topic_", "sent_"))]
        c = c.merge(nlp[["review_id", *nlp_cols]], on="review_id", how="left")
        # Leakage guard from build step: masked reviews must not carry NLP features.
        masked = c["review_text"].isna()
        c.loc[masked, nlp_cols] = np.nan
    return c, nlp_cols


def time_split(c: pd.DataFrame, test_frac: float = 0.25):
    cutoff = c["purchase_date"].quantile(1 - test_frac)
    return c[c["purchase_date"] < cutoff], c[c["purchase_date"] >= cutoff], cutoff


def _make_model(n_estimators: int) -> lgb.LGBMClassifier:
    # Strong regularisation: only ~600 positives in training.
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.02, num_leaves=15, min_child_samples=200,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
        min_split_gain=0.0, cat_smooth=50, random_state=RANDOM_SEED, verbose=-1,
    )


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    # Early stopping on the last 20% of the TRAIN period (time-ordered), never on the test set.
    tr = train.sort_values("purchase_date")
    cut = int(len(tr) * 0.8)
    inner_tr, inner_val = tr.iloc[:cut], tr.iloc[cut:]
    probe = _make_model(1500)
    probe.fit(
        inner_tr[feats], inner_tr["repurchased"],
        eval_set=[(inner_val[feats], inner_val["repurchased"])], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iter = max(int(probe.best_iteration_ or 100), 50)
    model = _make_model(best_iter)
    model.fit(train[feats], train["repurchased"])
    p = model.predict_proba(test[feats])[:, 1]
    y = test["repurchased"].values
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    metrics = {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "base_rate": round(float(y.mean()), 4),
        "lift_top10pct": round(float(y[top].mean() / max(y.mean(), 1e-9)), 3),
        "n_trees": best_iter,
    }
    return model, p, metrics


def survival(c_all: pd.DataFrame) -> None:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    c = c_all.copy()
    data_end = c["purchase_date"].max()
    censor = (data_end - c["purchase_date"]).dt.total_seconds() / 86400
    c["event"] = c["days_to_second"].notna().astype(int)
    c["duration"] = c["days_to_second"].fillna(censor).clip(lower=0.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = {
        "On time + 4-5 stars": (c["is_late"] == 0) & (c["review_score"] >= 4),
        "Late delivery": c["is_late"] == 1,
        "1-2 star review": c["review_score"] <= 2,
    }
    for label, mask in groups.items():
        km = KaplanMeierFitter().fit(c.loc[mask, "duration"], c.loc[mask, "event"], label=label)
        # Plot cumulative share who came back (1 - survival) for readability.
        (1 - km.survival_function_).plot(ax=ax)
    ax.set_xlim(0, 365)
    ax.set_xlabel("Days since first purchase")
    ax.set_ylabel("Share of customers who returned")
    ax.set_title("Time to second purchase by first experience (Kaplan-Meier)")
    fig.tight_layout()
    fig.savefig(FIG / "km_by_experience.png", dpi=130)
    plt.close(fig)

    cox_df = c[["duration", "event", "is_late", "review_score", "freight_ratio", "payment_value"]].dropna()
    cox_df = cox_df.assign(
        low_review=(cox_df["review_score"] <= 2).astype(int),
        log_payment=np.log1p(cox_df["payment_value"]),
        freight_ratio=cox_df["freight_ratio"].clip(upper=2),
    ).drop(columns=["review_score", "payment_value"])
    cph = CoxPHFitter(penalizer=0.01).fit(cox_df, "duration", "event")
    cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4).to_csv(
        ARTIFACTS / "survival_cox.csv"
    )


def main() -> None:
    c, nlp_cols = load_frame()
    train, test, cutoff = time_split(c)

    variants = {
        "A. order + delivery only": ORDER_FEATURES,
        "B. + review score": ORDER_FEATURES + REVIEW_SCORE_FEATURES,
    }
    if nlp_cols:
        variants["C. + review NLP (topics, sentiment)"] = ORDER_FEATURES + REVIEW_SCORE_FEATURES + nlp_cols

    rows, best = [], None
    for name, feats in variants.items():
        model, p, m = fit_eval(train, test, feats)
        rows.append({"variant": name, "n_features": len(feats), **m})
        best = (name, model, feats, p)
        print(name, m)
    ablation = pd.DataFrame(rows)
    ablation.to_csv(ARTIFACTS / "churn_ablation.csv", index=False)

    name, model, feats, p = best
    sample = test[feats].sample(min(5000, len(test)), random_state=RANDOM_SEED)
    sv = shap.TreeExplainer(model).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    imp.round(5).to_csv(ARTIFACTS / "shap_importance.csv", header=["mean_abs_shap"])
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG / "shap_summary.png", dpi=130)
    plt.close("all")

    # Score every observable customer with the final model (for the dashboard / Action Center).
    scores = c[["customer_unique_id", "review_id", "purchase_date", "repurchased"]].copy()
    scores["p_repurchase"] = model.predict_proba(c[feats])[:, 1]
    scores["split"] = np.where(c["purchase_date"] >= cutoff, "test", "train")
    scores.to_parquet(PROCESSED / "customer_scores.parquet", index=False)

    survival(pd.read_parquet(PROCESSED / "customers.parquet"))

    out = {
        "final_variant": name,
        "time_split_cutoff": str(cutoff.date()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "ablation": rows,
        "top_shap_features": imp.head(10).round(5).to_dict(),
    }
    (ARTIFACTS / "churn_metrics.json").write_text(encoding="utf-8", data=json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
