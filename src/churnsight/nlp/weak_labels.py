"""Weak supervision: Portuguese keyword labeling functions for review topics.

These rules are fast and transparent but brittle (they miss paraphrases and negations).
They produce *noisy training labels*; the fine-tuned transformer then learns to generalise
beyond the exact keywords. Both are evaluated on a separate hand-checked gold set.

Run:  python -m churnsight.nlp.weak_labels
"""

from __future__ import annotations

import re

import pandas as pd
from unidecode import unidecode

from churnsight.config import LABELS, PROCESSED

TOPICS = {
    "delivery_delay": "Delivery delayed / slow",
    "not_received": "Order not received",
    "fast_delivery": "Fast / on-time delivery",
    "defect_quality": "Defective / poor quality / damaged",
    "wrong_missing_item": "Wrong, different or missing item",
    "good_quality": "Good quality / as described",
    "support_refund": "Seller support, refund, return or cancellation",
    "praise_recommend": "Praise / would recommend",
}

# Patterns are applied to lower-cased, accent-stripped text.
PATTERNS: dict[str, list[str]] = {
    "delivery_delay": [
        r"\batras", r"\bdemor", r"fora do prazo", r"passou do prazo", r"ultrapass\w* o prazo",
        r"prazo (de entrega )?(nao|expirou|vencido)", r"muito tempo", r"\blent[ao]\b",
    ],
    "not_received": [
        r"(nao|ainda nao) (recebi|chegou|foi entregue|entregaram|veio)", r"\bnao recebemos\b",
        r"aguardando (o )?(produto|entrega|pedido)", r"cade (o|meu) (produto|pedido)",
        r"nunca chegou", r"\bextraviad",
    ],
    "fast_delivery": [
        r"antes do prazo", r"\bentrega (super |muito )?rapida", r"\brapide[zs]\b",
        r"chegou (bem |muito |super )?(rapido|antes|no prazo|dentro do prazo)",
        r"\bno prazo\b", r"dentro do prazo", r"\bentregue rapid", r"\bagilidade\b",
    ],
    "defect_quality": [
        r"\bdefeito", r"\bquebrad", r"nao funciona", r"parou de funcionar", r"\bdanificad",
        r"\brasgad", r"(ma|baixa|pessima) qualidade", r"qualidade (ruim|pessima|inferior|baixa)",
        r"\bamassad", r"\bvazand", r"\briscad", r"\bmanchad", r"\bfragil",
    ],
    "wrong_missing_item": [
        r"produto errado", r"veio errad", r"diferente d[oa]", r"nao (e|era) o (que|mesmo)",
        r"\bfaltando\b", r"\bfaltou\b", r"\bincomplet", r"(cor|tamanho|modelo) errad",
        r"outro produto", r"veio (apenas|so) (um|uma|1)", r"recebi (apenas|so) (um|uma|1)",
        r"\bfalta (de )?(uma|um|peca|item)",
    ],
    "good_quality": [
        r"(boa|otima|excelente|alta|bem feito|de) qualidade", r"conforme (o )?(descri|anunci|esperad)",
        r"de acordo com (a )?(descri|anunci|o anunciado)", r"igual (a|ao) (foto|anuncio|descri)",
        r"\bperfeit[oa]\b", r"\bbem acabad",
    ],
    "support_refund": [
        r"\bdevolu", r"\bdevolv", r"\breembols", r"\bestorn", r"\btroca\b", r"\bcancel",
        r"sem resposta", r"nao (me )?respond", r"\batendimento\b", r"\bsac\b", r"\breclama",
        r"entrei em contato", r"\bdinheiro de volta", r"\bprocon\b",
    ],
    "praise_recommend": [
        r"\brecomendo\b", r"\botim[oa]\b", r"\bexcelente\b", r"\badorei\b", r"\bamei\b",
        r"\bsatisfeit", r"\bparabens\b", r"\bnota (10|dez|mil)\b", r"\bmaravilhos",
        r"\bshow\b", r"\btop\b", r"\bgostei\b",
    ],
}

COMPILED = {t: re.compile("|".join(p)) for t, p in PATTERNS.items()}
NEGATED_PRAISE = re.compile(r"nao (recomendo|gostei|estou satisfeit|fiquei satisfeit)")


def topic_thresholds() -> dict[str, float]:
    """Per-label decision thresholds tuned by train_classifier (falls back to 0.5)."""
    import json

    from churnsight.config import ARTIFACTS

    p = ARTIFACTS / "topic_thresholds.json"
    th = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {t: float(th.get(t, 0.5)) for t in [*TOPICS, "negative"]}


def normalize(text: str) -> str:
    return unidecode(str(text)).lower()


def label_text(text: str) -> dict[str, int]:
    t = normalize(text)
    out = {topic: int(bool(rx.search(t))) for topic, rx in COMPILED.items()}
    if NEGATED_PRAISE.search(t):
        out["praise_recommend"] = 0
    return out


def apply(df: pd.DataFrame, col: str = "review_text") -> pd.DataFrame:
    labels = df[col].map(label_text).apply(pd.Series)
    return labels.astype(int)


def main() -> None:
    reviews = pd.read_parquet(PROCESSED / "reviews.parquet")
    weak = apply(reviews)
    out = pd.concat([reviews[["review_id", "review_text", "review_score"]], weak], axis=1)
    out.to_parquet(LABELS / "weak_labels.parquet", index=False)
    print("coverage (share with >=1 topic):", round(float(weak.any(axis=1).mean()), 3))
    print(weak.mean().round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
