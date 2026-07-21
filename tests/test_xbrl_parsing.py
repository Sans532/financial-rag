import httpx
import pytest
import respx

from src.agents.state import NumericClaim
from src.data.xbrl_client import XbrlClient, XbrlFact
from src.eval.faithfulness import (
    _match_period,
    _normalize_unit,
    faithfulness_score,
    verify_claim,
)


@respx.mock
def test_get_concept_parses_units_and_facts():
    payload = {
        "units": {
            "USD": [
                {
                    "val": 5000,
                    "start": "2024-01-01",
                    "end": "2024-03-31",
                    "fy": 2024,
                    "fp": "Q1",
                    "form": "10-Q",
                    "filed": "2024-05-01",
                },
                {
                    "val": 20000,
                    "start": "2023-10-01",
                    "end": "2024-09-30",
                    "fy": 2024,
                    "fp": "FY",
                    "form": "10-K",
                    "filed": "2024-11-01",
                },
            ]
        }
    }
    respx.get(
        "https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/NetIncomeLoss.json"
    ).mock(return_value=httpx.Response(200, json=payload))

    client = XbrlClient(user_agent="test test@example.com")
    facts = client.get_concept("0000320193", "NetIncomeLoss")

    assert len(facts) == 2
    assert facts[0].value == 5000
    assert facts[0].fiscal_period == "Q1"
    assert facts[1].form == "10-K"


@respx.mock
def test_get_concept_returns_empty_on_404():
    respx.get(
        "https://data.sec.gov/api/xbrl/companyconcept/CIK0000999999/us-gaap/NetIncomeLoss.json"
    ).mock(return_value=httpx.Response(404))

    client = XbrlClient(user_agent="test test@example.com")
    assert client.get_concept("0000999999", "NetIncomeLoss") == []


@respx.mock
def test_get_company_facts_parses_json():
    respx.get("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json").mock(
        return_value=httpx.Response(200, json={"cik": 320193, "facts": {}})
    )
    client = XbrlClient(user_agent="test test@example.com")
    data = client.get_company_facts("0000320193")
    assert data["cik"] == 320193


def _fact(value, fy, fp, form="10-Q", filed="2024-05-01"):
    return XbrlFact(
        tag="NetIncomeLoss",
        value=value,
        unit="USD",
        period_start=None,
        period_end="2024-03-31",
        fiscal_year=fy,
        fiscal_period=fp,
        form=form,
        filed=filed,
    )


def test_match_period_finds_matching_quarter():
    facts = [_fact(100, 2024, "Q1", filed="2024-05-01"), _fact(200, 2024, "Q2", filed="2024-08-01")]
    result = _match_period(facts, "Q1 2024")
    assert result is not None
    assert result.value == 100


def test_match_period_falls_back_to_fy_for_annual_period():
    facts = [_fact(100, 2024, "Q1"), _fact(999, 2024, "FY", filed="2024-12-01")]
    result = _match_period(facts, "FY2024")
    assert result is not None
    assert result.value == 999


def test_match_period_returns_none_without_year():
    facts = [_fact(100, 2024, "Q1")]
    assert _match_period(facts, "latest") is None


def test_normalize_unit_billions_and_millions():
    assert _normalize_unit(5.2, "USD_billions") == pytest.approx(5_200_000_000)
    assert _normalize_unit(120, "USD_millions") == pytest.approx(120_000_000)
    assert _normalize_unit(42, "%") == 42


def _claim(
    value, unit="USD", company="AAPL", metric="net_income", period="Q1 2024"
) -> NumericClaim:
    return NumericClaim(
        text=f"claim about {value}",
        metric=metric,
        value=value,
        unit=unit,
        company=company,
        period=period,
        source_span="",
    )


class _StubXbrlClient:
    def __init__(self, facts):
        self._facts = facts

    def get_metric(self, cik, metric):
        return self._facts


def test_verify_claim_confirmed_within_tolerance():
    claim = _claim(5_000_000_000, unit="USD")
    facts = [_fact(5_050_000_000, 2024, "Q1")]  # 1% off, within CONFIRMED_TOLERANCE_PCT
    result = verify_claim(claim, _StubXbrlClient(facts))
    assert result["status"] == "confirmed"
    assert result["xbrl_value"] == 5_050_000_000


def test_verify_claim_contradicted_outside_tolerance():
    claim = _claim(5_000_000_000, unit="USD")
    facts = [_fact(2_000_000_000, 2024, "Q1")]  # way off
    result = verify_claim(claim, _StubXbrlClient(facts))
    assert result["status"] == "contradicted"


def test_verify_claim_unverifiable_when_named_period_has_no_matching_fact():
    """If the claim names a specific period (e.g. the model mislabels a fiscal quarter)
    and no XBRL fact matches it, we must not silently fall back to comparing against a
    fact from a different period (e.g. an annual figure) — that would produce a false
    'contradicted' for what might be a perfectly correct claim."""
    claim = _claim(5_000_000_000, unit="USD", period="Q3 2024")
    facts = [_fact(5_000_000_000, 2024, "Q1"), _fact(20_000_000_000, 2024, "FY")]
    result = verify_claim(claim, _StubXbrlClient(facts))
    assert result["status"] == "unverifiable"
    assert result["xbrl_value"] is None


def test_verify_claim_uses_latest_fact_when_no_period_stated():
    claim = _claim(5_000_000_000, unit="USD", period="")
    facts = [_fact(5_000_000_000, 2024, "Q1", filed="2024-05-01")]
    result = verify_claim(claim, _StubXbrlClient(facts))
    assert result["status"] == "confirmed"


def test_verify_claim_unverifiable_when_no_facts():
    claim = _claim(5_000_000_000, unit="USD")
    result = verify_claim(claim, _StubXbrlClient([]))
    assert result["status"] == "unverifiable"
    assert result["xbrl_value"] is None


def test_verify_claim_unverifiable_for_unknown_company():
    claim = _claim(5_000_000_000, company="ZZZZ")
    result = verify_claim(claim, _StubXbrlClient([_fact(5_000_000_000, 2024, "Q1")]))
    assert result["status"] == "unverifiable"


def test_verify_claim_unverifiable_for_unknown_metric():
    claim = _claim(5_000_000_000, metric="not_a_real_metric")
    result = verify_claim(claim, _StubXbrlClient([_fact(5_000_000_000, 2024, "Q1")]))
    assert result["status"] == "unverifiable"


def test_verify_claim_handles_zero_fact_value():
    claim = _claim(100, unit="USD")
    facts = [_fact(0, 2024, "Q1")]
    result = verify_claim(claim, _StubXbrlClient(facts))
    assert result["status"] == "contradicted"
    assert result["delta_pct"] == 100.0


def test_faithfulness_score_no_claims_is_perfect():
    assert faithfulness_score([]) == 1.0


def test_faithfulness_score_mixed_results():
    results = [
        {"status": "confirmed"},
        {"status": "confirmed"},
        {"status": "contradicted"},
        {"status": "unverifiable"},
    ]
    assert faithfulness_score(results) == pytest.approx(0.5)
