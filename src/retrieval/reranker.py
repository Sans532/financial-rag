"""Cross-encoder reranking step applied after hybrid dense+BM25 fusion."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.retrieval.vector_store import ScoredChunk

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 8
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.record.text) for c in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [
            ScoredChunk(record=c.record, score=float(s), source="hybrid")
            for c, s in reranked[:top_k]
        ]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
