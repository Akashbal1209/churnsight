"""Central paths and constants so every module reads/writes the same places."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
# Deployments point this at the slim app bundle (data/app) built by scripts/export_app_data.py
PROCESSED = Path(os.getenv("CHURNSIGHT_PROCESSED_DIR", DATA / "processed"))
if not PROCESSED.is_absolute():
    PROCESSED = ROOT / PROCESSED
LABELS = DATA / "labels"
EVAL = DATA / "eval"
ARTIFACTS = ROOT / "artifacts"  # small, committed outputs (metrics, insights, figures)
MODELS = ROOT / "models"  # large, git-ignored model weights
POLICIES = ROOT / "docs" / "policies"

OLIST_DATASET = "olistbr/brazilian-ecommerce"

# Point-in-time design: we predict at PREDICTION_DAY days after the first purchase, using only
# information known by then (delivery, review), and ask whether the customer returns within the
# following REPURCHASE_WINDOW_DAYS. Day 30: ~95% of orders delivered, ~97% of reviews written.
PREDICTION_DAY = 30
REPURCHASE_WINDOW_DAYS = 180

RANDOM_SEED = 42

for _p in (RAW, INTERIM, PROCESSED, LABELS, EVAL, ARTIFACTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)
