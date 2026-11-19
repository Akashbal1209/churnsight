"""Action Center: prioritise unhappy customers and draft a policy-grounded reply.

Design rule: **rules decide money, the LLM writes words.**
- Priority and the eligible voucher come from deterministic, auditable rules.
- The LLM drafts the Portuguese reply from retrieved policy sections.
- Guardrails then check the draft: voucher must equal the rule-engine amount, every cited policy
  must be among the retrieved ones, any R$ amount in the text must be an allowed amount, and
  legal/safety keywords force escalation to a human.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from unidecode import unidecode

from churnsight.llm import client
from churnsight.nlp.weak_labels import TOPICS
from churnsight.rag.policies import policy_index

ALLOWED_VOUCHERS = {0, 15, 20, 30}
ESCALATION_TERMS = re.compile(r"procon|advogad|justica|processo|reclame aqui|policia|intoxica|alergi|queimad|ferid")

TOPIC_QUERY = {
    "delivery_delay": "order delivered late delay voucher",
    "not_received": "order not received investigation refund reship",
    "defect_quality": "damaged defective item replacement refund photos",
    "wrong_missing_item": "wrong item missing item partial refund",
    "support_refund": "return refund exchange seller did not answer escalation",
}


@dataclass
class Case:
    review_id: str
    review_text: str
    review_score: float
    topics: list[str]
    payment_value: float
    delay_days: float | None
    is_first_order: bool = True
    p_repurchase: float | None = None
    segment: str | None = None


@dataclass
class Suggestion:
    priority: str
    priority_reasons: list[str]
    eligible_voucher_brl: int
    voucher_reason: str
    policy_chunks: list[dict]
    reply_pt: str | None = None
    reply_en: str | None = None
    resolution: str | None = None
    escalate_to_human: bool = False
    escalation_reason: str | None = None
    policy_citations: list[str] = field(default_factory=list)
    guardrail_flags: list[str] = field(default_factory=list)
    llm_used: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def prioritise(case: Case) -> tuple[str, list[str]]:
    reasons = []
    negative = case.review_score is not None and case.review_score <= 2
    if negative:
        reasons.append(f"{int(case.review_score)}-star review")
    if "not_received" in case.topics:
        reasons.append("order not received")
    if "support_refund" in case.topics:
        reasons.append("unresolved return/refund or no seller answer")
    if case.payment_value >= 300:
        reasons.append(f"high order value (R${case.payment_value:.0f})")
    if case.delay_days and case.delay_days > 7:
        reasons.append(f"{case.delay_days:.0f} days late")
    if ESCALATION_TERMS.search(unidecode(case.review_text).lower()):
        reasons.append("legal/safety language")
        return "high", reasons
    if negative and ({"not_received", "support_refund"} & set(case.topics) or case.payment_value >= 300):
        return "high", reasons
    if negative:
        return "medium", reasons
    return "low", reasons or ["no complaint signal"]


def eligible_voucher(case: Case) -> tuple[int, str]:
    """Mirror of 06-retention-goodwill.md. Highest eligible single voucher."""
    negative = case.review_score is not None and case.review_score <= 2
    options = []
    if negative and case.payment_value >= 300:
        options.append((30, "R$30 goodwill: order >= R$300 with negative experience"))
    if case.delay_days and case.delay_days > 7:
        options.append((20, "R$20 delay voucher: more than 7 days late"))
    if negative and case.is_first_order:
        options.append((15, "R$15 goodwill: first order with 1-2 star review"))
    if not options:
        return 0, "not eligible for a voucher under the goodwill policy"
    return max(options)


def retrieve_policies(case: Case, k: int = 5) -> list[dict]:
    terms = [TOPIC_QUERY[t] for t in case.topics if t in TOPIC_QUERY] or ["customer complaint resolution"]
    query = " ; ".join(terms) + " ; goodwill voucher reply guidelines"
    hits = policy_index().search(query, k=k, mode="hybrid")
    chunks = hits[["chunk_id", "text"]].to_dict(orient="records")
    # Reply guidelines and goodwill rules are always in context.
    for must in ("08-reply-guidelines#language-and-tone", "06-retention-goodwill#allowed-offers"):
        if must not in {c["chunk_id"] for c in chunks}:
            row = policy_index().docs.set_index("chunk_id").loc[must]
            chunks.append({"chunk_id": must, "text": row["text"]})
    return chunks


SYSTEM = """You are a customer-care specialist at Aurora Marketplace, a Brazilian online marketplace.
You draft replies to customers who left reviews, strictly following the policy excerpts you are given.
Rules:
- Write the customer reply in Brazilian Portuguese, 3-6 sentences, signed "Equipe Aurora".
- Offer only resolutions stated in the policy excerpts. Never invent amounts, deadlines or benefits.
- The voucher amount is decided by the system and given to you; mention exactly that amount, or no voucher if it is 0.
- Cite the chunk_id of every policy excerpt you relied on.
- Set escalate_to_human=true if the case is not covered by the excerpts, or mentions legal action, consumer agencies or health/safety risk.
- Also give a faithful English translation of the reply for the analyst."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reply_pt": {"type": "string"},
        "reply_en": {"type": "string"},
        "resolution": {"type": "string", "description": "One-line internal next step for the support agent"},
        "voucher_brl": {"type": "integer", "enum": sorted(ALLOWED_VOUCHERS)},
        "escalate_to_human": {"type": "boolean"},
        "escalation_reason": {"type": "string"},
        "policy_citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply_pt", "reply_en", "resolution", "voucher_brl", "escalate_to_human",
                 "escalation_reason", "policy_citations"],
    "additionalProperties": False,
}


def check_guardrails(draft: dict, voucher: int, chunk_ids: set[str], review_text: str) -> list[str]:
    flags = []
    if draft["voucher_brl"] != voucher:
        flags.append(f"voucher mismatch: draft R${draft['voucher_brl']} vs policy R${voucher}")
    bad = [c for c in draft["policy_citations"] if c not in chunk_ids]
    if bad:
        flags.append(f"citations outside retrieved policies: {bad}")
    if not draft["policy_citations"]:
        flags.append("no policy cited")
    amounts = {int(a) for a in re.findall(r"R\$\s?(\d+)", draft["reply_pt"])}
    allowed_amounts = {voucher, 60}  # 60 = voucher minimum purchase in the policy
    if amounts - allowed_amounts:
        flags.append(f"unapproved R$ amounts in reply: {sorted(amounts - allowed_amounts)}")
    if ESCALATION_TERMS.search(unidecode(review_text).lower()) and not draft["escalate_to_human"]:
        flags.append("legal/safety language but not escalated")
    return flags


def suggest(case: Case, use_llm: bool | None = None) -> Suggestion:
    level, reasons = prioritise(case)
    voucher, voucher_reason = eligible_voucher(case)
    chunks = retrieve_policies(case)
    out = Suggestion(level, reasons, voucher, voucher_reason, chunks)

    if use_llm is None:
        use_llm = client.available()
    if not use_llm:
        return out

    topic_names = ", ".join(TOPICS[t] for t in case.topics) or "none detected"
    context = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)
    user = (
        f"POLICY EXCERPTS:\n{context}\n\n"
        f"CUSTOMER REVIEW ({int(case.review_score)} stars): \"{case.review_text}\"\n"
        f"Detected topics: {topic_names}\n"
        f"Order value: R${case.payment_value:.2f}; days late vs estimate: "
        f"{'n/a' if case.delay_days is None else f'{case.delay_days:.0f}'}\n"
        f"System-approved voucher: R${voucher} ({voucher_reason})\n\n"
        "Draft the reply and fill every field."
    )
    draft = client.complete_json(SYSTEM, user, SCHEMA, max_tokens=4000)
    out.llm_used = True
    out.reply_pt, out.reply_en, out.resolution = draft["reply_pt"], draft["reply_en"], draft["resolution"]
    out.escalate_to_human, out.escalation_reason = draft["escalate_to_human"], draft["escalation_reason"]
    out.policy_citations = draft["policy_citations"]
    out.guardrail_flags = check_guardrails(draft, voucher, {c["chunk_id"] for c in chunks}, case.review_text)
    if out.guardrail_flags:
        out.escalate_to_human = True
        out.escalation_reason = "guardrail: " + "; ".join(out.guardrail_flags)
    return out
