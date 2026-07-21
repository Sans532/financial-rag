"""Filing Retriever node — hybrid (dense + BM25) search over chunked SEC filings, reranked."""

from __future__ import annotations

import time
from functools import lru_cache

from src.agents.state import AgentState, RetrievedChunk
from src.logging_config import get_logger
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import get_vector_store

logger = get_logger(__name__)


@lru_cache
def get_hybrid_searcher() -> HybridSearcher:
    return HybridSearcher(
        vector_store=get_vector_store(),
        embedding_model=get_embedding_model(),
        reranker=get_reranker(),
    )


def filing_retriever_node(state: AgentState) -> dict:
    start = time.monotonic()
    searcher = get_hybrid_searcher()

    filing_subtasks = [s for s in state["subtasks"] if s["needs_filing"]]
    if not filing_subtasks:
        return {
            "trace": [
                {
                    "step": len(state["trace"]) + 1,
                    "agent": "filing_retriever",
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "summary": "skipped — no sub-task requires filing text",
                }
            ]
        }

    # On a retry, fold the verifier's hints into the query so retrieval targets the gap.
    hint_suffix = ""
    if state.get("retry_hints"):
        hint_suffix = " " + " ".join(state["retry_hints"])

    chunks: list[RetrievedChunk] = []
    errors: list[str] = []
    seen_keys: set[str] = set()

    for subtask in filing_subtasks:
        companies = subtask["companies"] or state["companies"]
        for company in companies:
            query = f"{subtask['description']}{hint_suffix}"
            try:
                results = searcher.search(query, company=company, final_k=5)
            except Exception:
                logger.error("filing_retrieval_failed", company=company, exc_info=True)
                errors.append(f"filing_retriever: search failed for {company}")
                continue
            for r in results:
                key = f"{r.record.accession_number}:{r.record.chunk_index}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                chunks.append(
                    RetrievedChunk(
                        content=r.record.text,
                        company=r.record.company,
                        filing_type=r.record.filing_type,
                        filing_date=r.record.filing_date,
                        section=r.record.section,
                        score=r.score,
                        source=r.source,
                        accession_number=r.record.accession_number,
                    )
                )

    return {
        "filing_chunks": chunks,
        "errors": errors,
        "trace": [
            {
                "step": len(state["trace"]) + 1,
                "agent": "filing_retriever",
                "duration_ms": (time.monotonic() - start) * 1000,
                "summary": (
                    f"retrieved {len(chunks)} chunk(s) across "
                    f"{len(filing_subtasks)} sub-task(s)"
                ),
            }
        ],
    }
