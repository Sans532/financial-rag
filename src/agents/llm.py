"""Shared Gemini client helpers used by the planner, verifier, and synthesizer nodes."""

from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import retry, retry_if_exception, stop_after_attempt

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

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


class _RateLimiter:
    """Client-side token-bucket pacing requests under a per-minute cap.

    Gemini's free tier caps gemini-3.1-flash-lite at 15 requests/minute — a limit this
    project's agent pipeline blows through easily (planner + synthesizer + claim
    extraction fire per question, doubling on every verifier retry). Paced below the cap
    with headroom rather than relying solely on retry-after-429, since bursts of 429s
    otherwise stall entire eval runs.
    """

    def __init__(self, max_per_minute: float = 8.0) -> None:
        self._min_interval = 60.0 / max_per_minute
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_rate_limiter = _RateLimiter()


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429


def _extract_retry_delay_seconds(exc: BaseException) -> float | None:
    """Pull the server-suggested cooldown out of a 429's RetryInfo, if present — more
    precise than a guessed backoff schedule since the API tells us exactly how long its
    own per-minute window has left."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    error_details = details.get("error", {}).get("details", [])
    for d in error_details:
        if isinstance(d, dict) and "retryDelay" in d:
            try:
                return float(str(d["retryDelay"]).rstrip("s"))
            except ValueError:
                return None
    return None


def _wait_for_rate_limit(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    delay = _extract_retry_delay_seconds(exc) if exc else None
    if delay is not None:
        return delay + 2.0  # small buffer past the API's own suggested cooldown
    return min(60.0, 3.0 * (2 ** (retry_state.attempt_number - 1)))  # fallback backoff


@retry(
    stop=stop_after_attempt(6),
    wait=_wait_for_rate_limit,
    retry=retry_if_exception(_is_rate_limit_error),
    reraise=True,
)
def _generate_content(
    model: str, contents: str, config: genai_types.GenerateContentConfig
) -> genai_types.GenerateContentResponse:
    _rate_limiter.wait()
    try:
        return get_client().models.generate_content(model=model, contents=contents, config=config)
    except genai_errors.ClientError as e:
        if getattr(e, "code", None) == 429:
            logger.warning("gemini_rate_limited")
        raise


def call_text(system: str, user: str, max_tokens: int = 2000) -> str:
    """Single-turn call returning the response text."""
    settings = get_settings()
    response = _generate_content(
        settings.gemini_model,
        user,
        genai_types.GenerateContentConfig(
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
    response = _generate_content(
        settings.gemini_model,
        user,
        genai_types.GenerateContentConfig(
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
