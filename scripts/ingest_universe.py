#!/usr/bin/env python
"""One-off script: ingest recent 10-K/10-Q filings for the full company universe.

Usage:
    python scripts/ingest_universe.py [--tickers AAPL,MSFT] [--filings-per-form 1]
"""

from __future__ import annotations

import argparse
import sys

from src.data.edgar_client import EdgarClient
from src.data.universe import COMPANIES, get_company
from src.logging_config import configure_logging, get_logger
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated subset of tickers")
    parser.add_argument("--filings-per-form", type=int, default=1)
    args = parser.parse_args()

    from src.data.ingest import ingest_company

    companies = (
        [get_company(t) for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else COMPANIES
    )

    embedder = get_embedding_model()
    store = get_vector_store()

    total_chunks = 0
    failures = []
    with EdgarClient() as edgar:
        for company in companies:
            logger.info("ingest_company_start", ticker=company.ticker)
            try:
                n = ingest_company(
                    company, edgar, embedder, store, filings_per_form=args.filings_per_form
                )
                total_chunks += n
                if n == 0:
                    failures.append(company.ticker)
            except Exception:
                logger.error("ingest_company_failed", ticker=company.ticker, exc_info=True)
                failures.append(company.ticker)

    logger.info("ingest_universe_complete", total_chunks=total_chunks, failures=failures)
    if failures:
        print(f"Completed with {len(failures)} failed companies: {failures}", file=sys.stderr)


if __name__ == "__main__":
    main()
