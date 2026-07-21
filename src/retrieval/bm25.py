"""Sparse (BM25) retrieval over the same chunk corpus stored in Qdrant."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import FilingChunkRecord, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index. Rebuilt from the vector store's payloads at query time
    (or once at ingestion and cached by the caller) — cheap for a few thousand chunks."""

    def __init__(self, records: list[FilingChunkRecord]) -> None:
        self.records = records
        self._corpus_tokens = [tokenize(r.text) for r in records]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            ScoredChunk(record=self.records[i], score=float(scores[i]), source="bm25")
            for i in ranked
            if scores[i] > 0
        ]
