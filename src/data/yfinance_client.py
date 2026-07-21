"""yfinance wrapper for price data and basic fundamentals. No API key required."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MarketSnapshot:
    ticker: str
    price: float | None
    market_cap: float | None
    pe_ratio: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    dividend_yield: float | None


class YFinanceClient:
    """Thin wrapper isolating yfinance so callers don't touch the third-party API directly."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_snapshot(self, ticker: str) -> MarketSnapshot | None:
        try:
            info: dict[str, Any] = yf.Ticker(ticker).get_info()
        except Exception:
            logger.error("yfinance_fetch_failed", ticker=ticker, exc_info=True)
            return None
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            logger.warning("yfinance_empty_info", ticker=ticker)
            return None
        return MarketSnapshot(
            ticker=ticker,
            price=info.get("currentPrice") or info.get("regularMarketPrice"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            dividend_yield=info.get("dividendYield"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get_recent_price_history(self, ticker: str, period: str = "3mo") -> list[dict[str, Any]]:
        try:
            hist = yf.Ticker(ticker).history(period=period)
        except Exception:
            logger.error("yfinance_history_failed", ticker=ticker, exc_info=True)
            return []
        if hist.empty:
            return []
        hist = hist.reset_index()
        return [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
            for _, row in hist.iterrows()
        ]
