"""Thin wrapper around the Anthropic Messages API.

- LLM_PROVIDER=anthropic -> live calls via the official SDK (credentials: ANTHROPIC_API_KEY or an
  `ant auth login` profile). LLM_PROVIDER=none (default) -> callers fall back to cached outputs,
  so the dashboard and tests run without any key.
- Structured outputs (output_config.format json_schema) guarantee parseable JSON.
- Server-side refusal fallback is enabled; refusals are surfaced as LLMRefusal.
- The system prompt is marked for prompt caching (stable prefix, reused across many calls).
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from churnsight.config import ROOT

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = ""
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    pass


class LLMRefusal(RuntimeError):
    pass


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "none").strip().lower()


def model() -> str:
    return os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL


def available() -> bool:
    if provider() != "anthropic":
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


_client = None


def _get_client():
    global _client
    if not available():
        raise LLMUnavailable("Set LLM_PROVIDER=anthropic (and credentials) to enable live LLM calls.")
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def _call(system: str, messages: list[dict], max_tokens: int, effort: str, output_format: dict | None):
    kwargs: dict[str, Any] = {
        "model": model(),
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
        "output_config": {"effort": effort, **({"format": output_format} if output_format else {})},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }
    resp = _get_client().beta.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        raise LLMRefusal(f"Model declined the request (category={cat}).")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return text, resp


def complete_text(system: str, user: str, max_tokens: int = 4000, effort: str = "medium") -> str:
    text, _ = _call(system, [{"role": "user", "content": user}], max_tokens, effort, None)
    return text


def complete_json(
    system: str, user: str, schema: dict, max_tokens: int = 8000, effort: str = "medium"
) -> dict:
    """Return a dict that validates against `schema` (JSON Schema with additionalProperties: false)."""
    fmt = {"type": "json_schema", "schema": schema}
    text, _ = _call(system, [{"role": "user", "content": user}], max_tokens, effort, fmt)
    return json.loads(text)
