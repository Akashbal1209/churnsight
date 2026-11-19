"""Fine-tune a Portuguese BERT (BERTimbau) for multi-label review topics + negativity.

Labels
  weak    Portuguese keyword rules on ~30k reviews (noisy, high coverage)
  silver  600 LLM-assisted labels (data/labels/silver.csv): 480 for training, 120 held out
  gold    250 LLM-annotated reviews, 50 per star rating (data/labels/gold.csv): evaluation ONLY
"negative" is review_score <= 2 everywhere.

Two-stage training
  stage 1  fine-tune BERTimbau on the weak labels (learns the domain from lots of noisy data)
  stage 2  continue fine-tuning on the 480 silver reviews (corrects systematic keyword errors)
  then     per-label decision thresholds tuned on the 120 silver holdout (never seen in training)

Compared on gold: keyword rules, TF-IDF + LogReg (weak / weak+silver), BERTimbau stage 1,
stage 1+2, and stage 1+2 with tuned thresholds.

Outputs
  models/review_classifier/              final weights (git-ignored)
  artifacts/topic_thresholds.json        per-label thresholds used downstream
  data/processed/review_nlp.parquet      topic_* and sent_neg probabilities for every review
  artifacts/nlp_comparison.csv, artifacts/nlp_metrics.json

Run:  python -m churnsight.nlp.train_classifier
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from churnsight.config import ARTIFACTS, LABELS, MODELS, PROCESSED, RANDOM_SEED
from churnsight.nlp.weak_labels import TOPICS, apply

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
LABEL_COLS = list(TOPICS) + ["negative"]
MAX_LEN = 128
OUT_DIR = MODELS / "review_classifier"
SILVER_HOLDOUT = 120
STAGE2_EPOCHS = 4  # fixed in advance; not selected on gold
THRESHOLD_GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)


class TextDS(Dataset):
    def __init__(self, enc, labels=None):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.float)
        return item


def scores(y_true: np.ndarray, y_pred: np.ndarray, cols: list[str]) -> dict:
    n_topics = len(TOPICS)  # first columns are topics, last is "negative"
    per = {c: round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 3) for i, c in enumerate(cols)}
    f1 = lambda a, b: round(float(f1_score(a, b, average="macro", zero_division=0)), 4)  # noqa: E731
    return {
        "macro_f1": f1(y_true, y_pred),
        "topic_macro_f1": f1(y_true[:, :n_topics], y_pred[:, :n_topics]),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_label_f1": per,
    }


def predict_probs(model, tok, texts: list[str], batch: int = 128) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            out.append(torch.sigmoid(model(**enc).logits).float().cpu().numpy())
    return np.vstack(out)


def tfidf_probs(tr_df: pd.DataFrame, texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    lr = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0))
    lr.fit(vec.fit_transform(tr_df["review_text"]), tr_df[LABEL_COLS].values)
    return lr.predict_proba(vec.transform(texts))


def stage1(tr_df: pd.DataFrame, val_df: pd.DataFrame):
    torch.manual_seed(RANDOM_SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABEL_COLS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABEL_COLS)), label2id={c: i for i, c in enumerate(LABEL_COLS)},
    )
    enc = lambda texts: tok(list(texts), truncation=True, max_length=MAX_LEN)  # noqa: E731
    train_ds = TextDS(enc(tr_df["review_text"]), tr_df[LABEL_COLS].values.astype(np.float32))
    val_ds = TextDS(enc(val_df["review_text"]), val_df[LABEL_COLS].values.astype(np.float32))
    epochs, batch = 2, 32
    total_steps = -(-len(train_ds) // batch) * epochs
    args = TrainingArguments(
        output_dir=str(MODELS / "_ckpt"), num_train_epochs=epochs, per_device_train_batch_size=batch,
        per_device_eval_batch_size=128, learning_rate=3e-5, warmup_steps=int(0.06 * total_steps),
        weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=200,
        fp16=torch.cuda.is_available(), report_to=[], seed=RANDOM_SEED,
    )
    Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tok)).train()
    return model, tok


def stage2(model, tok, silver_tr: pd.DataFrame, texts: list[str]):
    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ds = TextDS(tok(texts, truncation=True, max_length=MAX_LEN), silver_tr[LABEL_COLS].values.astype(np.float32))
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=DataCollatorWithPadding(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    for _ in range(STAGE2_EPOCHS):
        model.train()
        for b in dl:
            loss = model(**{k: v.to(device) for k, v in b.items()}).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    return model


def tune_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    th = []
    for j in range(y.shape[1]):
        th.append(max(THRESHOLD_GRID, key=lambda t: (f1_score(y[:, j], (p[:, j] >= t).astype(int), zero_division=0), -abs(t - 0.5))))
    return np.array(th)


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    weak = pd.read_parquet(LABELS / "weak_labels.parquet")
    weak["negative"] = (weak["review_score"] <= 2).astype(int)
    text_of = weak.set_index("review_id")["review_text"]
    gold = pd.read_csv(LABELS / "gold.csv")
    silver = pd.read_csv(LABELS / "silver.csv").sample(frac=1, random_state=RANDOM_SEED)
    silver_hold, silver_tr = silver.iloc[:SILVER_HOLDOUT], silver.iloc[SILVER_HOLDOUT:]
    gold_texts = text_of.loc[gold["review_id"]].tolist()
    y_gold = gold[LABEL_COLS].values

    # Stage-1 data: weak labels, excluding every gold and silver review.
    pool = weak[~weak["review_id"].isin(set(gold["review_id"]) | set(silver["review_id"]))]
    pool = pool.sample(frac=1, random_state=RANDOM_SEED)
    n_val = int(len(pool) * 0.1)
    val_df, tr_df = pool.iloc[:n_val], pool.iloc[n_val:]

    results = {}
    rules = apply(pd.DataFrame({"review_text": gold_texts}))
    rules["negative"] = 0  # keywords cannot detect negativity
    results["1. Keyword rules"] = scores(y_gold, rules[LABEL_COLS].values, LABEL_COLS)

    results["2a. TF-IDF + LogReg (weak)"] = scores(y_gold, (tfidf_probs(tr_df, gold_texts) >= 0.5).astype(int), LABEL_COLS)
    cols = ["review_text", *LABEL_COLS]
    silver_rows = silver_tr.assign(review_text=text_of.loc[silver_tr["review_id"]].values)[cols]
    tr_mix = pd.concat([tr_df[cols]] + [silver_rows] * 8)  # oversample the 480 silver reviews
    results["2b. TF-IDF + LogReg (weak + silver)"] = scores(y_gold, (tfidf_probs(tr_mix, gold_texts) >= 0.5).astype(int), LABEL_COLS)

    model, tok = stage1(tr_df, val_df)
    results["3a. BERTimbau stage 1 (weak)"] = scores(y_gold, (predict_probs(model, tok, gold_texts) >= 0.5).astype(int), LABEL_COLS)

    model = stage2(model, tok, silver_tr, text_of.loc[silver_tr["review_id"]].tolist())
    p_gold = predict_probs(model, tok, gold_texts)
    results["3b. BERTimbau stage 1+2 (weak -> silver)"] = scores(y_gold, (p_gold >= 0.5).astype(int), LABEL_COLS)

    p_hold = predict_probs(model, tok, text_of.loc[silver_hold["review_id"]].tolist())
    th = tune_thresholds(silver_hold[LABEL_COLS].values, p_hold)
    results["3c. + thresholds tuned on silver holdout"] = scores(y_gold, (p_gold >= th).astype(int), LABEL_COLS)
    for k, v in results.items():
        print(f"{k:45s} macro {v['macro_f1']:.4f}  topics {v['topic_macro_f1']:.4f}")

    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    thresholds = {c: float(t) for c, t in zip(LABEL_COLS, th)}
    (ARTIFACTS / "topic_thresholds.json").write_text(encoding="utf-8", data=json.dumps(thresholds, indent=2))

    # Score every review for downstream use (churn features, impact, dashboard, RAG metadata).
    all_reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    probs = predict_probs(model, tok, all_reviews["review_text"].tolist())
    nlp = pd.DataFrame(probs, columns=[f"topic_{c}" for c in TOPICS] + ["sent_neg"])
    nlp.insert(0, "review_id", all_reviews["review_id"].values)
    nlp.to_parquet(PROCESSED / "review_nlp.parquet", index=False)

    rows = [{"model": name, **{k: v for k, v in r.items() if k != "per_label_f1"}} for name, r in results.items()]
    pd.DataFrame(rows).to_csv(ARTIFACTS / "nlp_comparison.csv", index=False)
    meta = {
        "base_model": BASE_MODEL, "n_stage1_train": int(len(tr_df)), "n_stage1_val": int(len(val_df)),
        "n_silver_train": int(len(silver_tr)), "n_silver_holdout": int(len(silver_hold)), "n_gold": int(len(gold)),
        "stage2_epochs": STAGE2_EPOCHS, "thresholds": thresholds, "results": results,
    }
    (ARTIFACTS / "nlp_metrics.json").write_text(encoding="utf-8", data=json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
