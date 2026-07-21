"""SEC EDGAR XBRL CompanyFacts client — the structured ground truth used by the verifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.data.edgar_client import EdgarRateLimiter
from src.logging_config import get_logger

logger = get_logger(__name__)

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
COMPANY_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
)

# Commonly-referenced us-gaap tags, keyed by a friendly metric name used elsewhere
# in this system (planner sub-tasks, eval set, verifier claim extraction).
METRIC_TO_TAGS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
    ],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "operating_expenses": ["OperatingExpenses", "CostsAndExpenses"],
}

_rate_limiter = EdgarRateLimiter()


@dataclass
class XbrlFact:
    tag: str
    value: float
    unit: str
    period_start: str | None
    period_end: str
    fiscal_year: int
    fiscal_period: str  # "FY", "Q1", "Q2", "Q3", "Q4"
    form: str
    filed: str


class XbrlClient:
    """Rate-limited client over data.sec.gov/api/xbrl/."""

    def __init__(self, user_agent: str | None = None) -> None:
        settings = get_settings()
        self.user_agent = user_agent or settings.sec_user_agent
        self._client = httpx.Client(headers={"User-Agent": self.user_agent}, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> XbrlClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response | None:
        _rate_limiter.wait()
        resp = self._client.get(url)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    def get_company_facts(self, cik: str) -> dict[str, Any] | None:
        url = COMPANY_FACTS_URL.format(cik=cik.zfill(10))
        resp = self._get(url)
        return resp.json() if resp is not None else None

    def get_concept(
        self, cik: str, tag: str, taxonomy: str = "us-gaap"
    ) -> list[XbrlFact]:
        """Fetch all reported values for a single XBRL tag (e.g. 'NetIncomeLoss')."""
        url = COMPANY_CONCEPT_URL.format(cik=cik.zfill(10), taxonomy=taxonomy, tag=tag)
        resp = self._get(url)
        if resp is None:
            return []
        data = resp.json()
        facts: list[XbrlFact] = []
        for unit, entries in data.get("units", {}).items():
            for e in entries:
                facts.append(
                    XbrlFact(
                        tag=tag,
                        value=float(e["val"]),
                        unit=unit,
                        period_start=e.get("start"),
                        period_end=e["end"],
                        fiscal_year=e.get("fy", 0),
                        fiscal_period=e.get("fp", ""),
                        form=e.get("form", ""),
                        filed=e.get("filed", ""),
                    )
                )
        return facts

    def get_metric(self, cik: str, metric: str) -> list[XbrlFact]:
        """Fetch facts for a friendly metric name, trying tag aliases in order."""
        tags = METRIC_TO_TAGS.get(metric)
        if not tags:
            logger.warning("xbrl_unknown_metric", metric=metric)
            return []
        for tag in tags:
            facts = self.get_concept(cik, tag)
            if facts:
                return facts
        return []

    def latest_value(
        self, cik: str, metric: str, form: str | None = None
    ) -> XbrlFact | None:
        """Most-recently-filed fact for a metric, optionally restricted to a form type."""
        facts = self.get_metric(cik, metric)
        if form:
            facts = [f for f in facts if f.form == form]
        if not facts:
            return None
        return max(facts, key=lambda f: f.filed)
