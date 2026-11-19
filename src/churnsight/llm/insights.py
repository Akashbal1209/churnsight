"""Turn complaint clusters into business-ready insights with an LLM, and verify the evidence.

For each cluster the LLM sees keywords + representative reviews + operational stats and returns:
name, summary, severity, owner team, recommended action, and verbatim evidence quotes.
Every quote is checked against the cluster's reviews (accent/case-insensitive substring match);
quotes that do not appear are dropped and counted as hallucinations.

Output: artifacts/cluster_insights.json
Run:  python -m churnsight.llm.insights     (requires LLM_PROVIDER=anthropic)
"""

from __future__ import annotations

import json

import pandas as pd
from unidecode import unidecode

from churnsight.config import ARTIFACTS, PROCESSED
from churnsight.llm import client

SYSTEM = """You turn clusters of Brazilian e-commerce complaints (Portuguese) into concise insights for
a marketplace's operations and product teams. Write in English. Be concrete and grounded only in the
reviews and statistics given. Evidence quotes must be copied VERBATIM (in Portuguese) from the reviews."""

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "3-6 word issue name"},
        "summary": {"type": "string", "description": "2 sentences: what goes wrong and for whom"},
        "severity": {"type": "string", "enum": ["high", "medium", "low"]},
        "owner_team": {"type": "string", "enum": ["Logistics", "Seller Quality", "Customer Support", "Catalog", "Payments", "Product"]},
        "recommended_action": {"type": "string"},
        "evidence_quotes_pt": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "summary", "severity", "owner_team", "recommended_action", "evidence_quotes_pt"],
    "additionalProperties": False,
}


def _norm(s: str) -> str:
    return " ".join(unidecode(s).lower().split())


def verify_quotes(quotes: list[str], texts: list[str]) -> tuple[list[str], list[str]]:
    corpus = [_norm(t) for t in texts]
    ok, bad = [], []
    for q in quotes:
        nq = _norm(q).strip(" .\"'")
        (ok if nq and any(nq in t for t in corpus) else bad).append(q)
    return ok, bad


def main() -> None:
    meta = json.loads((ARTIFACTS / "complaint_clusters.json").read_text(encoding="utf-8"))
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    assign = pd.read_parquet(PROCESSED / "complaint_clusters.parquet").merge(reviews, on="review_id")

    results, n_quotes, n_bad = [], 0, 0
    for c in meta["clusters"]:
        sub = assign[assign["cluster"] == c["cluster"]]
        sample = list(dict.fromkeys(c["representative_reviews"] + sub["review_text"].sample(
            min(15, len(sub)), random_state=0).tolist()))[:18]
        user = (
            f"Cluster size: {c['size']} complaints; avg review score {c['avg_review_score']}; "
            f"late-delivery share {c['late_share']:.0%}; top categories {c['top_categories']}.\n"
            f"Keywords: {', '.join(c['keywords'])}\n\nReviews:\n" + "\n".join(f"- {t}" for t in sample)
        )
        out = client.complete_json(SYSTEM, user, SCHEMA, max_tokens=3000)
        ok, bad = verify_quotes(out["evidence_quotes_pt"], sub["review_text"].tolist())
        n_quotes += len(out["evidence_quotes_pt"])
        n_bad += len(bad)
        results.append({**c, **out, "evidence_quotes_pt": ok, "rejected_quotes": bad})
        print(f"[{c['cluster']}] {out['name']} ({out['severity']}, {out['owner_team']})")

    payload = {
        "generated_by": client.model(),
        "quote_verification": {"quotes": n_quotes, "not_found_in_reviews": n_bad},
        "clusters": results,
    }
    (ARTIFACTS / "cluster_insights.json").write_text(encoding="utf-8", data=json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if not client.available():
        raise SystemExit("LLM disabled: set LLM_PROVIDER=anthropic in .env to generate cluster insights.")
    main()
