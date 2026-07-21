"""Retrieval precision metric: did the retriever return chunks from the filing section
that actually contains the answer, per a small hand-labeled set (data/retrieval_labels.json)?
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel

from src.config import get_settings
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.vector_store import ScoredChunk


class AcceptableSource(BaseModel):
    filing_type: str
    section_contains: str


class RetrievalLabel(BaseModel):
    question_id: str
    companies: list[str]
    acceptable: list[AcceptableSource]


def load_retrieval_labels() -> list[RetrievalLabel]:
    settings = get_settings()
    with open(settings.retrieval_labels_path) as f:
        raw = json.load(f)
    return [RetrievalLabel.model_validate(item) for item in raw]


@dataclass
class RetrievalPrecisionResult:
    question_id: str
    hit: bool
    top_k: int


def _matches(chunk_results: list[ScoredChunk], label: RetrievalLabel) -> bool:
    for r in chunk_results:
        for acc in label.acceptable:
            if (
                r.record.filing_type == acc.filing_type
                and acc.section_contains.lower() in r.record.section.lower()
                and r.record.company in label.companies
            ):
                return True
    return False


def evaluate_retrieval_precision(
    searcher: HybridSearcher,
    questions_by_id: dict[str, str],
    top_k: int = 5,
) -> list[RetrievalPrecisionResult]:
    """For each labeled question, run hybrid search per company and check whether any
    top-k result lands in an acceptable (filing_type, section) bucket."""
    labels = load_retrieval_labels()
    results: list[RetrievalPrecisionResult] = []

    for label in labels:
        query = questions_by_id.get(label.question_id)
        if query is None:
            continue
        merged: list[ScoredChunk] = []
        for company in label.companies:
            merged.extend(searcher.search(query, company=company, final_k=top_k))
        hit = _matches(merged, label)
        results.append(
            RetrievalPrecisionResult(question_id=label.question_id, hit=hit, top_k=top_k)
        )

    return results


def precision_score(results: list[RetrievalPrecisionResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit) / len(results)
