"""LLM provider abstraction: Gemini (REST) primary, Anthropic optional.

Keys come exclusively from the environment via
:func:`pitchiq.core.env.get_secret` — never from configs — and are never
logged. ``provider: auto`` picks the first provider with a key present;
callers must handle ``None`` (no provider / call failed) by falling back to
the deterministic template.
"""

from __future__ import annotations

import logging

import requests

from pitchiq.config import LLMConfig
from pitchiq.core.env import get_secret

log = logging.getLogger(__name__)

GEMINI_DEFAULT_MODEL = "gemini-flash-latest"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def resolve_provider(cfg: LLMConfig) -> str:
    """'gemini' | 'anthropic' | 'none' given config + available keys."""
    if cfg.provider == "none":
        return "none"
    if cfg.provider in ("gemini", "auto") and get_secret("GEMINI_API_KEY"):
        return "gemini"
    if cfg.provider in ("anthropic", "auto") and get_secret("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def generate(system: str, user: str, cfg: LLMConfig) -> str | None:
    """One grounded generation call. Returns text or None on any failure."""
    provider = resolve_provider(cfg)
    if provider == "gemini":
        return _gemini(system, user, cfg)
    if provider == "anthropic":
        return _anthropic(system, user, cfg)
    return None


def _gemini(system: str, user: str, cfg: LLMConfig) -> str | None:
    key = get_secret("GEMINI_API_KEY")
    model = cfg.model if cfg.model not in ("auto", "") and "claude" not in cfg.model \
        else GEMINI_DEFAULT_MODEL
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": cfg.temperature,
            # flash models burn budget on internal reasoning; disable it and
            # give ample headroom so reports don't truncate mid-sentence
            "maxOutputTokens": max(cfg.max_tokens, 6000),
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                GEMINI_URL.format(model=model),
                headers={"Content-Type": "application/json", "X-goog-api-key": key},
                json=body, timeout=90,
            )
            if resp.status_code in (429, 500, 503) and attempt < 2:
                import time

                time.sleep(3 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            cand = data["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            # skip internal 'thought' parts some models emit
            text = "\n".join(p.get("text", "") for p in parts
                             if not p.get("thought")).strip()
            if cand.get("finishReason") == "MAX_TOKENS":
                log.warning("gemini hit MAX_TOKENS; output may be truncated")
            if text:
                log.info("gemini generation ok (model=%s, %d chars)", model, len(text))
                return text
            return None
        except Exception as exc:
            log.warning("gemini call failed (attempt %d): %s", attempt + 1,
                        _safe_err(exc))
            if attempt == 2:
                return None
    return None


def _anthropic(system: str, user: str, cfg: LLMConfig) -> str | None:
    try:
        import anthropic

        model = cfg.model if cfg.model not in ("auto", "") and "gemini" not in cfg.model \
            else ANTHROPIC_DEFAULT_MODEL
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        msg = client.messages.create(
            model=model, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception as exc:
        log.warning("anthropic call failed: %s", _safe_err(exc))
        return None


def _safe_err(exc: Exception) -> str:
    """Error text with anything key-shaped stripped (never leak secrets)."""
    s = str(exc)
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        v = get_secret(name)
        if v:
            s = s.replace(v, "***")
    return s[:300]
