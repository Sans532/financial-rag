"""Hybrid dense + BM25 retrieval with reciprocal-rank fusion, then cross-encoder reranking."""

from __future__ import annotations

from src.logging_config import get_logger
from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import ScoredChunk, VectorStore

logger = get_logger(__name__)

RRF_K = 60  # standard reciprocal-rank-fusion smoothing constant


def reciprocal_rank_fusion(
    dense: list[ScoredChunk], sparse: list[ScoredChunk], k: int = RRF_K
) -> list[ScoredChunk]:
    """Merge two ranked lists by reciprocal rank rather than raw score — avoids having to
    normalize BM25 scores (unbounded) against cosine similarity (bounded [0,1])."""
    fused: dict[str, float] = {}
    by_key: dict[str, ScoredChunk] = {}

    def key_of(c: ScoredChunk) -> str:
        return f"{c.record.accession_number}:{c.record.chunk_index}"

    for rank, chunk in enumerate(dense):
        k_ = key_of(chunk)
        fused[k_] = fused.get(k_, 0.0) + 1.0 / (k + rank + 1)
        by_key[k_] = chunk

    for rank, chunk in enumerate(sparse):
        k_ = key_of(chunk)
        fused[k_] = fused.get(k_, 0.0) + 1.0 / (k + rank + 1)
        by_key.setdefault(k_, chunk)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [
        ScoredChunk(record=by_key[k_].record, score=score, source="hybrid")
        for k_, score in ordered
    ]


class HybridSearcher:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = reranker

    def search(
        self,
        query: str,
        company: str | None = None,
        filing_type: str | None = None,
        dense_k: int = 20,
        sparse_k: int = 20,
        final_k: int = 8,
        use_reranker: bool = True,
    ) -> list[ScoredChunk]:
        query_vector = self.embedding_model.embed_one(query)
        dense_results = self.vector_store.search(
            query_vector, top_k=dense_k, company=company, filing_type=filing_type
        )

        # BM25 index is rebuilt from the (small, filtered) corpus each call — fine at this scale.
        corpus = self.vector_store.scroll_all(company=company)
        if filing_type:
            corpus = [r for r in corpus if r.filing_type == filing_type]
        bm25_index = BM25Index(corpus)
        sparse_results = bm25_index.search(query, top_k=sparse_k)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        if use_reranker and self.reranker is not None and fused:
            return self.reranker.rerank(query, fused, top_k=final_k)
        return fused[:final_k]
