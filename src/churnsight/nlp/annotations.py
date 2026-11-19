"""Turn annotation files into label tables (one column per label).  Run: python -m churnsight.nlp.annotations

  data/labels/gold_annotations.txt   + gold_to_label.csv   -> gold.csv    (evaluation only)
  data/labels/silver_annotations.txt + silver_to_label.csv -> silver.csv  (extra training data)
"""

import pandas as pd

from churnsight.config import LABELS

CODES = {
    "D": "delivery_delay", "N": "not_received", "F": "fast_delivery", "Q": "defect_quality",
    "W": "wrong_missing_item", "G": "good_quality", "S": "support_refund", "P": "praise_recommend",
}


def build(name: str) -> pd.DataFrame:
    sample = pd.read_csv(LABELS / f"{name}_to_label.csv")
    ann = {}
    for line in (LABELS / f"{name}_annotations.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        idx, codes = line.split()
        ann[int(idx)] = codes
    assert set(ann) == set(range(len(sample))), f"every {name} review must be annotated"
    out = sample[["review_id", "review_score"]].copy()
    for code, col in CODES.items():
        out[col] = [int(code in ann[i]) for i in range(len(sample))]
    out["negative"] = (out["review_score"] <= 2).astype(int)
    out.to_csv(LABELS / f"{name}.csv", index=False)
    return out


if __name__ == "__main__":
    gold, silver = build("gold"), build("silver")
    assert not set(gold["review_id"]) & set(silver["review_id"]), "gold and silver must not overlap"
    print(pd.DataFrame({"gold": gold[list(CODES.values())].sum(), "silver": silver[list(CODES.values())].sum()}))
