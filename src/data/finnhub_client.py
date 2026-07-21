"""Finnhub free-tier client for company news. Requires FINNHUB_API_KEY (email signup only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

BASE_URL = "https://finnhub.io/api/v1"


@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    url: str
    datetime: str  # ISO date


class FinnhubClient:
    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.finnhub_api_key
        self._client = httpx.Client(timeout=15.0)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FinnhubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    def get_company_news(self, ticker: str, days_back: int = 30, limit: int = 10) -> list[NewsItem]:
        if not self.enabled:
            logger.warning("finnhub_disabled_no_key")
            return []
        today = date.today()
        params = {
            "symbol": ticker,
            "from": (today - timedelta(days=days_back)).isoformat(),
            "to": today.isoformat(),
            "token": self.api_key,
        }
        try:
            resp = self._client.get(f"{BASE_URL}/company-news", params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError:
            logger.error("finnhub_news_failed", ticker=ticker)
            return []
        items = resp.json() or []
        out = []
        for item in items[:limit]:
            out.append(
                NewsItem(
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    datetime=date.fromtimestamp(item["datetime"]).isoformat()
                    if item.get("datetime")
                    else "",
                )
            )
        return out
