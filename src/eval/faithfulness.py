"""Automated faithfulness metric: extract numeric claims from an answer and cross-check
them against SEC XBRL structured data (the ground truth).

Shared by the live Verifier agent (src/agents/verifier.py) and the offline eval harness
(scripts/run_eval.py) so "verified live" and "verified in eval" use identical logic.
"""

from __future__ import annotations

import re

from src.agents.llm import call_json
from src.agents.state import NumericClaim, VerificationResult
from src.data.universe import TICKER_TO_COMPANY
from src.data.xbrl_client import METRIC_TO_TAGS, XbrlClient
from src.logging_config import get_logger

logger = get_logger(__name__)

CLAIM_EXTRACTION_SYSTEM_PROMPT = f"""Extract every numeric financial claim from the given text \
(dollar amounts, percentages, ratios, per-share figures). For each claim, output an object with:

- "text": the exact sentence or clause containing the claim
- "metric": one of {list(METRIC_TO_TAGS.keys())} (pick the closest match; if none fit, use "other")
- "value": the numeric value as a plain float (convert e.g. "$5.2 billion" -> 5200000000, \
"12%" -> 12)
- "unit": "USD", "USD_millions", "USD_billions", "%", "ratio", or "per_share"
- "company": the ticker the claim is about
- "period": the fiscal period referenced, e.g. "FY2024", "Q3 2024", "latest", or "" if unclear

Respond with ONLY a JSON array. If there are no numeric claims, respond with [].
"""

# Relative tolerance for comparing a claimed value to the XBRL ground truth. Generous enough
# to absorb rounding ("$5.2B" vs. an exact $5,183,000,000 fact) without masking real errors.
CONFIRMED_TOLERANCE_PCT = 2.0

_PERIOD_QUARTER_RE = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
# No \b before the year: periods like "FY2024" have no word-boundary between the
# letters and the digits (both are \w), so a boundary-anchored pattern would miss them.
_PERIOD_YEAR_RE = re.compile(r"(19|20)\d{2}")


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    if not text.strip():
        return []
    try:
        raw = call_json(CLAIM_EXTRACTION_SYSTEM_PROMPT, text, max_tokens=2000)
    except Exception:
        logger.error("claim_extraction_failed", exc_info=True)
        return []

    claims: list[NumericClaim] = []
    for item in raw if isinstance(raw, list) else []:
        try:
            claims.append(
                NumericClaim(
                    text=str(item["text"]),
                    metric=str(item.get("metric", "other")),
                    value=float(item["value"]),
                    unit=str(item.get("unit", "")),
                    company=str(item.get("company", "")).upper(),
                    period=str(item.get("period", "")),
                    source_span=str(item["text"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return claims


def _normalize_unit(value: float, unit: str) -> float:
    unit = unit.lower()
    if "billion" in unit:
        return value * 1_000_000_000
    if "million" in unit:
        return value * 1_000_000
    return value


def _match_period(facts: list, period: str):
    year_match = _PERIOD_YEAR_RE.search(period)
    quarter_match = _PERIOD_QUARTER_RE.search(period)
    if not year_match:
        return None
    year = int(year_match.group(0))
    candidates = [f for f in facts if f.fiscal_year == year]
    if quarter_match:
        fp = f"Q{quarter_match.group(1)}"
        quarter_candidates = [f for f in candidates if f.fiscal_period == fp]
        if quarter_candidates:
            return max(quarter_candidates, key=lambda f: f.filed)
    elif candidates:
        fy_candidates = [f for f in candidates if f.fiscal_period == "FY"]
        return max(fy_candidates or candidates, key=lambda f: f.filed)
    return None


def verify_claim(claim: NumericClaim, xbrl_client: XbrlClient) -> VerificationResult:
    company = TICKER_TO_COMPANY.get(claim["company"])
    if company is None or claim["metric"] not in METRIC_TO_TAGS:
        return VerificationResult(
            claim=claim, xbrl_value=None, xbrl_tag=None, status="unverifiable", delta_pct=None
        )

    facts = xbrl_client.get_metric(company.cik, claim["metric"])
    if not facts:
        return VerificationResult(
            claim=claim, xbrl_value=None, xbrl_tag=None, status="unverifiable", delta_pct=None
        )

    period = claim["period"].strip().lower()
    if period and period != "latest":
        # A specific period was named — only compare against a fact for that exact
        # period. Falling back to "whatever was filed most recently" here risks comparing
        # a quarterly claim against an annual figure (or vice versa) whenever the model's
        # stated period doesn't match XBRL's period labels (e.g. mixing up a company's
        # fiscal quarter with the calendar quarter) — that's an apples-to-oranges
        # mismatch on our side, not necessarily an error in the claim, so we report
        # "unverifiable" rather than a false "contradicted".
        fact = _match_period(facts, claim["period"])
        if fact is None:
            return VerificationResult(
                claim=claim, xbrl_value=None, xbrl_tag=None, status="unverifiable", delta_pct=None
            )
    else:
        fact = max(facts, key=lambda f: f.filed)

    claimed_value = _normalize_unit(claim["value"], claim["unit"])

    if fact.value == 0:
        delta_pct = 100.0 if claimed_value != 0 else 0.0
    else:
        delta_pct = abs(claimed_value - fact.value) / abs(fact.value) * 100

    status = "confirmed" if delta_pct <= CONFIRMED_TOLERANCE_PCT else "contradicted"

    return VerificationResult(
        claim=claim,
        xbrl_value=fact.value,
        xbrl_tag=fact.tag,
        status=status,
        delta_pct=round(delta_pct, 2),
    )


def verify_claims(claims: list[NumericClaim], xbrl_client: XbrlClient) -> list[VerificationResult]:
    return [verify_claim(c, xbrl_client) for c in claims]


def faithfulness_score(results: list[VerificationResult]) -> float:
    """Fraction of numeric claims that were verifiably correct. Unverifiable claims count
    against the score (they could not be checked, so they aren't "verifiably correct")."""
    if not results:
        return 1.0  # no numeric claims made -> nothing to be unfaithful about
    confirmed = sum(1 for r in results if r["status"] == "confirmed")
    return confirmed / len(results)
