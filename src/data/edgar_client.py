"""Rate-limited SEC EDGAR client.

Respects SEC's fair-use policy: max 10 req/sec, descriptive User-Agent header.
https://www.sec.gov/os/webmaster-faq#developers
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"


class EdgarRateLimiter:
    """Simple token-bucket limiter enforcing SEC's 10 req/sec fair-use cap."""

    def __init__(self, max_per_second: float = 8.0) -> None:
        self._min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_rate_limiter = EdgarRateLimiter()


@dataclass
class FilingRef:
    accession_number: str  # e.g. "0000320193-24-000123"
    form: str  # "10-K", "10-Q", "8-K"
    filing_date: str
    report_date: str
    primary_document: str
    cik: str

    @property
    def accession_nodash(self) -> str:
        return self.accession_number.replace("-", "")


class EdgarClient:
    """Thin, rate-limited wrapper over the SEC EDGAR submissions + Archives APIs."""

    def __init__(self, user_agent: str | None = None) -> None:
        settings = get_settings()
        self.user_agent = user_agent or settings.sec_user_agent
        if "@" not in self.user_agent:
            logger.warning(
                "sec_user_agent_missing_contact",
                msg="SEC requires a contact email in the User-Agent header",
            )
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        _rate_limiter.wait()
        resp = self._client.get(url, **kwargs)
        if resp.status_code == 429:
            logger.warning("edgar_rate_limited", url=url)
            raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    def get_submissions(self, cik: str) -> dict[str, Any]:
        """Fetch the full filing history for a company (data.sec.gov/submissions)."""
        cik_padded = cik.zfill(10)
        url = SUBMISSIONS_URL.format(cik=cik_padded)
        try:
            resp = self._get(url)
        except httpx.HTTPStatusError:
            logger.error("edgar_submissions_failed", cik=cik)
            raise
        return resp.json()

    def list_filings(
        self, cik: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 8
    ) -> list[FilingRef]:
        """Return the most recent filings of the given forms for a company."""
        data = self.get_submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        forms_list = recent.get("form", [])
        out: list[FilingRef] = []
        for i, form in enumerate(forms_list):
            if form not in forms:
                continue
            out.append(
                FilingRef(
                    accession_number=recent["accessionNumber"][i],
                    form=form,
                    filing_date=recent["filingDate"][i],
                    report_date=recent.get("reportDate", [""] * len(forms_list))[i],
                    primary_document=recent["primaryDocument"][i],
                    cik=cik.zfill(10),
                )
            )
            if len(out) >= limit:
                break
        return out

    def get_filing_document(self, filing: FilingRef) -> str:
        """Download the primary HTML document for a filing."""
        cik_int = int(filing.cik)
        url = ARCHIVES_URL.format(
            cik_int=cik_int,
            accession_nodash=filing.accession_nodash,
            filename=filing.primary_document,
        )
        try:
            resp = self._get(url)
        except httpx.HTTPStatusError:
            logger.error("edgar_document_fetch_failed", url=url)
            raise
        return resp.text
