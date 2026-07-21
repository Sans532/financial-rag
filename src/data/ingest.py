"""Ingestion pipeline: fetch filings from EDGAR -> chunk -> embed -> upsert into Qdrant.

Errors on any single filing/company are caught and logged so one bad document
doesn't abort a full-universe ingestion run.
"""

from __future__ import annotations

from src.data.edgar_client import EdgarClient, FilingRef
from src.data.universe import Company
from src.logging_config import get_logger
from src.retrieval.chunking import chunk_filing
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import FilingChunkRecord, VectorStore

logger = get_logger(__name__)


def ingest_filing(
    company: Company,
    filing: FilingRef,
    edgar: EdgarClient,
    embedder: EmbeddingModel,
    store: VectorStore,
) -> int:
    """Ingest a single filing. Returns the number of chunks written (0 on failure)."""
    try:
        html = edgar.get_filing_document(filing)
    except Exception:
        logger.error(
            "ingest_fetch_failed", ticker=company.ticker, accession=filing.accession_number,
            exc_info=True,
        )
        return 0

    try:
        chunks = chunk_filing(html, filing_type=filing.form)
    except Exception:
        logger.error(
            "ingest_chunk_failed", ticker=company.ticker, accession=filing.accession_number,
            exc_info=True,
        )
        return 0

    if not chunks:
        logger.warning(
            "ingest_no_chunks", ticker=company.ticker, accession=filing.accession_number
        )
        return 0

    records = [
        FilingChunkRecord(
            text=c.text,
            company=company.ticker,
            filing_type=filing.form,
            filing_date=filing.filing_date,
            section=c.section,
            chunk_index=c.chunk_index,
            accession_number=filing.accession_number,
        )
        for c in chunks
    ]

    try:
        vectors = embedder.embed([r.text for r in records])
        store.ensure_collection(vector_size=embedder.dimension)
        store.upsert(records, vectors)
    except Exception:
        logger.error(
            "ingest_upsert_failed", ticker=company.ticker, accession=filing.accession_number,
            exc_info=True,
        )
        return 0

    logger.info(
        "ingest_filing_complete",
        ticker=company.ticker,
        form=filing.form,
        accession=filing.accession_number,
        chunks=len(records),
    )
    return len(records)


def ingest_company(
    company: Company,
    edgar: EdgarClient,
    embedder: EmbeddingModel,
    store: VectorStore,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    filings_per_form: int = 2,
) -> int:
    """Ingest the most recent filings for one company. Returns total chunks written."""
    try:
        filings = edgar.list_filings(company.cik, forms=forms, limit=filings_per_form * len(forms))
    except Exception:
        logger.error("ingest_list_filings_failed", ticker=company.ticker, exc_info=True)
        return 0

    total = 0
    for filing in filings:
        total += ingest_filing(company, filing, edgar, embedder, store)
    return total
