"""Fast unit tests for the rule-based parts (no model downloads, no data files)."""

import pandas as pd
import pytest

from churnsight.analytics.impact import detect_spikes
from churnsight.llm.actions import Case, check_guardrails, eligible_voucher, prioritise
from churnsight.llm.insights import verify_quotes
from churnsight.nlp.weak_labels import label_text


# ---------- weak labels ----------
@pytest.mark.parametrize(
    "text, topic",
    [
        ("Ainda não recebi o produto", "not_received"),
        ("Chegou antes do prazo, recomendo!", "fast_delivery"),
        ("Produto veio com defeito e quebrado", "defect_quality"),
        ("Comprei 2 mas veio apenas um", "wrong_missing_item"),
        ("Solicitei a devolução e não tive resposta", "support_refund"),
    ],
)
def test_weak_labels_hit(text, topic):
    assert label_text(text)[topic] == 1


def test_negated_praise_is_not_praise():
    assert label_text("Não recomendo, péssimo")["praise_recommend"] == 0


# ---------- trend spike detection ----------
def test_spike_detected_when_injected():
    months = [f"2018-0{i}" for i in range(1, 9)]
    base = [0.05, 0.051, 0.049, 0.05, 0.052, 0.048, 0.05]
    df = pd.DataFrame({"delivery_delay": base + [0.12], "fast_delivery": [0.3] * 8}, index=months)
    alerts = detect_spikes(df, lookback=6, z_thresh=2.0)
    assert [a["topic"] for a in alerts] == ["delivery_delay"]
    assert alerts[0]["month"] == "2018-08"


def test_no_spike_on_flat_series():
    df = pd.DataFrame({"x": [0.1, 0.11, 0.1, 0.09, 0.1, 0.11, 0.1, 0.1]})
    assert detect_spikes(df) == []


# ---------- action center rules ----------
def _case(**kw):
    base = dict(review_id="r", review_text="produto não chegou", review_score=1, topics=["not_received"],
                payment_value=120.0, delay_days=10.0)
    base.update(kw)
    return Case(**base)


def test_voucher_picks_highest_eligible():
    assert eligible_voucher(_case(payment_value=350))[0] == 30
    assert eligible_voucher(_case(payment_value=120, delay_days=10))[0] == 20
    assert eligible_voucher(_case(payment_value=120, delay_days=None))[0] == 15
    assert eligible_voucher(_case(review_score=5, delay_days=None))[0] == 0


def test_priority_high_for_not_received_and_legal_language():
    assert prioritise(_case())[0] == "high"
    assert prioritise(_case(topics=[], review_text="vou ao Procon"))[0] == "high"
    assert prioritise(_case(topics=["defect_quality"], payment_value=50))[0] == "medium"


def test_guardrails_flag_wrong_voucher_and_unknown_citation():
    draft = {"voucher_brl": 30, "policy_citations": ["made-up#x"], "reply_pt": "Oferecemos R$30 e R$100.",
             "escalate_to_human": False}
    flags = check_guardrails(draft, voucher=15, chunk_ids={"06-retention-goodwill#allowed-offers"},
                             review_text="não chegou")
    assert any("voucher mismatch" in f for f in flags)
    assert any("citations outside" in f for f in flags)
    assert any("unapproved R$" in f for f in flags)


def test_guardrails_pass_clean_draft():
    draft = {"voucher_brl": 15, "policy_citations": ["06-retention-goodwill#allowed-offers"],
             "reply_pt": "Cupom de R$15 para compras acima de R$60.", "escalate_to_human": False}
    assert check_guardrails(draft, 15, {"06-retention-goodwill#allowed-offers"}, "não chegou") == []


# ---------- insight quote verification ----------
def test_quote_verification_is_accent_insensitive():
    ok, bad = verify_quotes(["nao recebi o produto", "texto inventado"], ["Ainda NÃO recebi o produto!"])
    assert ok == ["nao recebi o produto"] and bad == ["texto inventado"]
