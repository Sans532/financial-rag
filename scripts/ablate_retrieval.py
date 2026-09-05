#!/usr/bin/env python
"""Retrieval ablation: dense-only vs BM25-only vs hybrid (RRF) vs hybrid + cross-encoder.

Answers "what did hybrid retrieval and reranking actually buy us?" with numbers rather
than assertion. Pure retrieval — no LLM calls, so this costs nothing and needs no
GOOGLE_API_KEY, only an ingested corpus.

Each arm gets the same candidate budget (dense_k / sparse_k) and returns the same final_k,
so the only variable is how candidates are selected and ordered.

Usage:
    python scripts/ablate_retrieval.py [--final-k 5] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from src.eval.eval_set import load_eval_set
from src.eval.retrieval_hit_rate import RetrievalLabel, load_retrieval_labels
from src.logging_config import configure_logging, get_logger
from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.hybrid_search import reciprocal_rank_fusion
from src.retrieval.reranker import get_reranker
from src.retrieval.vector_store import ScoredChunk, get_vector_store

logger = get_logger(__name__)

ARMS = ["dense_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_rerank"]


def _is_acceptable(chunk: ScoredChunk, label: RetrievalLabel) -> bool:
    r = chunk.record
    return any(
        r.filing_type == acc.filing_type
        and acc.section_contains.lower() in r.section.lower()
        and r.company in label.companies
        for acc in label.acceptable
    )


def _first_hit_rank(per_company: list[list[ScoredChunk]], label: RetrievalLabel) -> int | None:
    """Best 1-indexed rank at which acceptable evidence appears, minimised across the
    companies named in the question.

    Ranking is per company, not over the concatenation of every company's results: the
    live retriever issues one search per company, so acceptable evidence at rank 1 for
    the third company is a rank-1 result, not a rank-11 one.
    """
    best: int | None = None
    for results in per_company:
        for i, chunk in enumerate(results):
            if _is_acceptable(chunk, label):
                best = i + 1 if best is None else min(best, i + 1)
                break
    return best


@dataclass
class ArmResult:
    question_id: str
    arm: str
    first_hit_rank: int | None


def run_arms(
    query: str,
    label: RetrievalLabel,
    store,
    embedder,
    reranker,
    dense_k: int,
    sparse_k: int,
    final_k: int,
) -> dict[str, list[list[ScoredChunk]]]:
    """Run all four arms for one question, merging per-company results as the live
    retriever does (one search per company named in the question)."""
    per_arm: dict[str, list[list[ScoredChunk]]] = {a: [] for a in ARMS}
    qvec = embedder.embed_one(query)

    for company in label.companies:
        dense = store.search(qvec, top_k=dense_k, company=company)

        corpus = store.scroll_all(company=company)
        sparse = BM25Index(corpus).search(query, top_k=sparse_k)

        fused = reciprocal_rank_fusion(dense, sparse)
        reranked = reranker.rerank(query, fused, top_k=final_k) if fused else []

        per_arm["dense_only"].append(dense[:final_k])
        per_arm["bm25_only"].append(sparse[:final_k])
        per_arm["hybrid_rrf"].append(fused[:final_k])
        per_arm["hybrid_rrf_rerank"].append(reranked)

    return per_arm


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--sparse-k", type=int, default=20)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    labels = load_retrieval_labels()
    questions = {q.id: q.question for q in load_eval_set()}
    categories = {q.id: q.category for q in load_eval_set()}

    store = get_vector_store()
    embedder = get_embedding_model()
    reranker = get_reranker()

    rows: list[ArmResult] = []
    skipped: list[str] = []

    for label in labels:
        query = questions.get(label.question_id)
        if query is None:
            skipped.append(label.question_id)
            continue
        per_arm = run_arms(
            query, label, store, embedder, reranker,
            args.dense_k, args.sparse_k, args.final_k,
        )
        for arm, results in per_arm.items():
            rows.append(
                ArmResult(
                    question_id=label.question_id,
                    arm=arm,
                    first_hit_rank=_first_hit_rank(results, label),
                )
            )
        logger.info(
            "ablation_question_done",
            question_id=label.question_id,
            ranks={a: _first_hit_rank(r, label) for a, r in per_arm.items()},
        )

    n = len({r.question_id for r in rows})

    def hit_at(arm: str, k: int) -> float:
        rs = [r for r in rows if r.arm == arm]
        hits = sum(1 for r in rs if r.first_hit_rank is not None and r.first_hit_rank <= k)
        return hits / len(rs)

    def mrr(arm: str) -> float:
        rs = [r for r in rows if r.arm == arm]
        return sum(1.0 / r.first_hit_rank for r in rs if r.first_hit_rank) / len(rs)

    print(f"\nRetrieval ablation — {n} labeled questions, final_k={args.final_k}")
    if skipped:
        print(f"(skipped {len(skipped)} labels with no matching eval question: {skipped})")
    print(f"\n{'Arm':<22}{'hit@1':>9}{'hit@3':>9}{f'hit@{args.final_k}':>9}{'MRR':>9}")
    for arm in ARMS:
        print(
            f"{arm:<22}{hit_at(arm,1):>8.1%}{hit_at(arm,3):>9.1%}"
            f"{hit_at(arm,args.final_k):>9.1%}{mrr(arm):>9.3f}"
        )

    # Per-category breakdown at final_k, to show where each arm earns its keep.
    cats = sorted({categories.get(r.question_id, "?") for r in rows})
    print(f"\n{'Category':<16}" + "".join(f"{a[:14]:>16}" for a in ARMS))
    for cat in cats:
        ids = {r.question_id for r in rows if categories.get(r.question_id) == cat}
        cells = []
        for arm in ARMS:
            rs = [r for r in rows if r.arm == arm and r.question_id in ids]
            hits = sum(1 for r in rs if r.first_hit_rank and r.first_hit_rank <= args.final_k)
            cells.append(f"{hits/len(rs):>15.0%}" if rs else f"{'n/a':>15}")
        print(f"{cat:<16}" + "".join(f"{c} " for c in cells))

    if args.json:
        payload = {
            "config": {"final_k": args.final_k, "dense_k": args.dense_k, "sparse_k": args.sparse_k},
            "n_questions": n,
            "summary": {
                arm: {
                    "hit@1": hit_at(arm, 1),
                    "hit@3": hit_at(arm, 3),
                    f"hit@{args.final_k}": hit_at(arm, args.final_k),
                    "mrr": mrr(arm),
                }
                for arm in ARMS
            },
            "per_question": [
                {"question_id": r.question_id, "arm": r.arm, "first_hit_rank": r.first_hit_rank}
                for r in rows
            ],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
