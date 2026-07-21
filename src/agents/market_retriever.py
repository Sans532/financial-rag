"""Market/News Retriever node — recent news (Finnhub) and price/fundamentals (yfinance)."""

from __future__ import annotations

import time
from dataclasses import asdict

from src.agents.state import AgentState
from src.data.finnhub_client import FinnhubClient
from src.data.yfinance_client import YFinanceClient
from src.logging_config import get_logger

logger = get_logger(__name__)


def market_retriever_node(state: AgentState) -> dict:
    start = time.monotonic()
    market_subtasks = [s for s in state["subtasks"] if s["needs_market_data"]]
    if not market_subtasks:
        return {
            "trace": [
                {
                    "step": len(state["trace"]) + 1,
                    "agent": "market_retriever",
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "summary": "skipped — no sub-task requires market data",
                }
            ]
        }

    companies: set[str] = set()
    for s in market_subtasks:
        companies.update(s["companies"] or state["companies"])

    news: list[dict] = []
    fundamentals: list[dict] = []
    errors: list[str] = []

    yf_client = YFinanceClient()
    with FinnhubClient() as finnhub_client:
        for ticker in companies:
            try:
                snapshot = yf_client.get_snapshot(ticker)
                if snapshot:
                    fundamentals.append(asdict(snapshot))
            except Exception:
                logger.error("market_fundamentals_failed", ticker=ticker, exc_info=True)
                errors.append(f"market_retriever: fundamentals failed for {ticker}")

            if not finnhub_client.enabled:
                continue
            try:
                items = finnhub_client.get_company_news(ticker, days_back=30, limit=5)
                for item in items:
                    news.append({"ticker": ticker, **asdict(item)})
            except Exception:
                logger.error("market_news_failed", ticker=ticker, exc_info=True)
                errors.append(f"market_retriever: news failed for {ticker}")

    return {
        "market_news": news,
        "fundamentals": fundamentals,
        "errors": errors,
        "trace": [
            {
                "step": len(state["trace"]) + 1,
                "agent": "market_retriever",
                "duration_ms": (time.monotonic() - start) * 1000,
                "summary": f"fetched {len(fundamentals)} fundamentals, {len(news)} news item(s)",
            }
        ],
    }
