"""Qdrant-backed persistent vector store for chunked filings.

Real vector DB (not in-memory) so retrieval/citation metadata (company, filing
type, date, section) survives restarts and can be filtered on at query time.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FilingChunkRecord:
    text: str
    company: str
    filing_type: str
    filing_date: str
    section: str
    chunk_index: int
    accession_number: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    record: FilingChunkRecord
    score: float
    source: str  # "dense" | "bm25" | "hybrid"


class VectorStore:
    def __init__(self, collection_name: str | None = None) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection
        if settings.qdrant_local_path:
            self._client = QdrantClient(path=settings.qdrant_local_path)
        else:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )

    def ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name in existing:
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )
        logger.info("qdrant_collection_created", collection=self.collection_name)

    def upsert(self, records: list[FilingChunkRecord], vectors: list[list[float]]) -> None:
        if not records:
            return
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": r.text,
                    "company": r.company,
                    "filing_type": r.filing_type,
                    "filing_date": r.filing_date,
                    "section": r.section,
                    "chunk_index": r.chunk_index,
                    "accession_number": r.accession_number,
                    **r.metadata,
                },
            )
            for r, vector in zip(records, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        company: str | None = None,
        filing_type: str | None = None,
    ) -> list[ScoredChunk]:
        must: list[qmodels.FieldCondition] = []
        if company:
            must.append(
                qmodels.FieldCondition(key="company", match=qmodels.MatchValue(value=company))
            )
        if filing_type:
            must.append(
                qmodels.FieldCondition(
                    key="filing_type", match=qmodels.MatchValue(value=filing_type)
                )
            )
        query_filter = qmodels.Filter(must=must) if must else None

        try:
            hits = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            ).points
        except Exception:
            logger.error("qdrant_search_failed", exc_info=True)
            return []

        return [
            ScoredChunk(
                record=FilingChunkRecord(
                    text=h.payload["text"],
                    company=h.payload["company"],
                    filing_type=h.payload["filing_type"],
                    filing_date=h.payload["filing_date"],
                    section=h.payload["section"],
                    chunk_index=h.payload["chunk_index"],
                    accession_number=h.payload["accession_number"],
                ),
                score=h.score,
                source="dense",
            )
            for h in hits
        ]

    def scroll_all(self, company: str | None = None) -> list[FilingChunkRecord]:
        """Fetch all stored chunks (optionally filtered by company) — backs the BM25 index."""
        must = []
        if company:
            must.append(
                qmodels.FieldCondition(key="company", match=qmodels.MatchValue(value=company))
            )
        query_filter = qmodels.Filter(must=must) if must else None

        records: list[FilingChunkRecord] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                records.append(
                    FilingChunkRecord(
                        text=p.payload["text"],
                        company=p.payload["company"],
                        filing_type=p.payload["filing_type"],
                        filing_date=p.payload["filing_date"],
                        section=p.payload["section"],
                        chunk_index=p.payload["chunk_index"],
                        accession_number=p.payload["accession_number"],
                    )
                )
            if offset is None:
                break
        return records

    def collection_exists(self) -> bool:
        existing = {c.name for c in self._client.get_collections().collections}
        return self.collection_name in existing


@lru_cache
def get_vector_store() -> VectorStore:
    """Process-wide singleton. Required when QDRANT_LOCAL_PATH is set: Qdrant's embedded
    (on-disk, no server) mode only allows one open client per storage path at a time, so
    every caller in a process must share one QdrantClient instance rather than each
    constructing their own."""
    return VectorStore()
