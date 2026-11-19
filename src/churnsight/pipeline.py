"""Run the whole ChurnSight pipeline in order.

    python -m churnsight.pipeline              # everything (GPU recommended for the classifier)
    python -m churnsight.pipeline --skip-train # reuse an existing fine-tuned classifier
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time

from churnsight.config import MODELS, RAW

STEPS = [
    ("Build clean tables", "churnsight.data.build"),
    ("Cohorts + EDA", "churnsight.analytics.cohorts"),
    ("Segmentation", "churnsight.analytics.segments"),
    ("Weak labels", "churnsight.nlp.weak_labels"),
    ("Gold + silver label tables", "churnsight.nlp.annotations"),
    ("Fine-tune topic classifier", "churnsight.nlp.train_classifier"),
    ("Embeddings + complaint clusters", "churnsight.nlp.clusters"),
    ("Retrieval evaluation", "churnsight.rag.evaluate"),
    ("Repurchase model + SHAP + survival", "churnsight.analytics.churn"),
    ("Issue impact + trends", "churnsight.analytics.impact"),
    ("Action queue", "churnsight.llm.queue"),
]

OLIST_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"


def ensure_data() -> None:
    target = RAW / "olist"
    if (target / "olist_orders_dataset.csv").exists():
        return
    import io
    import zipfile
    import urllib.request

    print("Downloading Olist dataset (~45 MB)...")
    with urllib.request.urlopen(OLIST_URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    ensure_data()
    for name, module in STEPS:
        if args.skip_train and module.endswith("train_classifier") and (MODELS / "review_classifier").exists():
            print(f"-- skip {name}")
            continue
        print(f"\n== {name} ({module})")
        t0 = time.time()
        importlib.import_module(module)  # fail fast on import errors
        subprocess.run([sys.executable, "-m", module], check=True)
        print(f"   done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
