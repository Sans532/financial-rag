"""Shared Gemini client helpers used by the planner, verifier, and synthesizer nodes."""

from __future__ import annotations

import json
import re
import threading
from functools import lru_cache

from google import genai
from google.genai import types as genai_types

from src.config import get_settings

_usage_lock = threading.Lock()
_usage_log: list[dict] = []


def get_usage_log() -> list[dict]:
    with _usage_lock:
        return list(_usage_log)


def reset_usage_log() -> None:
    with _usage_lock:
        _usage_log.clear()


@lru_cache
def get_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key or None)


def _log_usage(model: str, response: genai_types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    with _usage_lock:
        _usage_log.append(
            {
                "model": model,
                "input_tokens": (usage.prompt_token_count if usage else 0) or 0,
                "output_tokens": (usage.candidates_token_count if usage else 0) or 0,
            }
        )


def call_text(system: str, user: str, max_tokens: int = 2000) -> str:
    """Single-turn call returning the response text."""
    settings = get_settings()
    client = get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        ),
    )
    _log_usage(settings.gemini_model, response)
    return response.text or ""


_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def call_json(system: str, user: str, max_tokens: int = 2000) -> dict | list:
    """Call the model in JSON mode and parse the response.

    Uses Gemini's `response_mime_type="application/json"` for a reliable structured
    response; falls back to regex-extracting the first {...} or [...] block in case a
    truncated or unusual response slips through without being valid JSON on its own.
    """
    settings = get_settings()
    client = get_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    _log_usage(settings.gemini_model, response)
    text = response.text or ""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"Model response did not contain parseable JSON: {text[:200]!r}")
    return json.loads(match.group(0))
