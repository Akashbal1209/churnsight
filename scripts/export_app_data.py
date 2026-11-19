"""Export the slim data bundle the deployed dashboard needs (data/app/, committed to git).

The full pipeline outputs (hundreds of MB, raw data, model weights) stay local. The deployed app
only needs: review texts + topic probabilities (for search and the Action Center) and the action
queue. Source data: Olist Brazilian E-commerce dataset, CC BY-NC-SA 4.0 (attribution in README).

Run:  python scripts/export_app_data.py
"""

import shutil

import pandas as pd

from churnsight.config import DATA, PROCESSED

APP = DATA / "app"
APP.mkdir(parents=True, exist_ok=True)

reviews = pd.read_parquet(PROCESSED / "reviews.parquet")[
    ["review_id", "review_score", "order_month", "category", "customer_state", "review_text"]
]
reviews.to_parquet(APP / "reviews.parquet", index=False)

nlp = pd.read_parquet(PROCESSED / "review_nlp.parquet")
nlp[nlp.columns[1:]] = nlp[nlp.columns[1:]].astype("float16")
nlp.to_parquet(APP / "review_nlp.parquet", index=False)

shutil.copy(PROCESSED / "action_queue.parquet", APP / "action_queue.parquet")

for p in sorted(APP.glob("*")):
    print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB")
