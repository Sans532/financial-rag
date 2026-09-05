#!/usr/bin/env python
"""Regenerate the eval report end-to-end.

Runs every question in data/eval_set.json through the full agent graph, checks
retrieval precision against data/retrieval_labels.json, and writes eval_report.md.

Usage:
    python scripts/run_eval.py [--limit N]
"""

from __future__ import annotations

import argparse

from src.agents.filing_retriever import get_hybrid_searcher
from src.agents.graph import get_graph
from src.agents.llm import get_usage_log, reset_usage_log
from src.agents.state import initial_state
from src.config import get_settings
from src.eval.eval_set import EvalQuestion, load_eval_set
from src.eval.latency import estimate_cost_usd, summarize_latency
from src.eval.report import EvalReport, QuestionResult, render_markdown
from src.eval.retrieval_hit_rate import evaluate_retrieval_hit_rate, hit_rate_score
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def _concept_coverage(answer: str, required_concepts: list[str]) -> float:
    if not required_concepts:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for concept in required_concepts if concept.lower() in answer_lower)
    return hits / len(required_concepts)


def run_question(question: EvalQuestion) -> tuple[QuestionResult, list[dict]]:
    graph = get_graph()
    state = initial_state(question.question, question.companies)

    try:
        result = graph.invoke(state)
    except Exception:
        logger.error("eval_question_failed", question_id=question.id, exc_info=True)
        return (
            QuestionResult(
                id=question.id,
                category=question.category,
                difficulty=question.difficulty,
                question=question.question,
                faithfulness_score=None,
                num_claims=0,
                concept_coverage=None,
                latency=summarize_latency([]),
                errors=[f"pipeline invocation failed for {question.id}"],
            ),
            [],
        )

    verification = result["verification_results"]
    faithfulness = None
    if verification:
        confirmed = sum(1 for v in verification if v["status"] == "confirmed")
        faithfulness = confirmed / len(verification)

    concept_coverage = None
    if question.expected.type == "qualitative":
        concept_coverage = _concept_coverage(
            result["final_answer"] or result["draft_answer"], question.expected.required_concepts
        )

    qr = QuestionResult(
        id=question.id,
        category=question.category,
        difficulty=question.difficulty,
        question=question.question,
        faithfulness_score=faithfulness,
        num_claims=len(verification),
        concept_coverage=concept_coverage,
        latency=summarize_latency(result["trace"]),
        errors=result["errors"],
    )
    return qr, verification


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    args = parser.parse_args()

    settings = get_settings()
    questions = load_eval_set()
    if args.limit:
        questions = questions[: args.limit]

    reset_usage_log()

    question_results: list[QuestionResult] = []
    all_verification: list[dict] = []
    for q in questions:
        logger.info("eval_running_question", question_id=q.id, category=q.category)
        qr, verification = run_question(q)
        question_results.append(qr)
        all_verification.extend(verification)

    total_claims = len(all_verification)
    confirmed = sum(1 for v in all_verification if v["status"] == "confirmed")
    global_faithfulness = confirmed / total_claims if total_claims else 0.0

    # Retrieval precision — separate, cheaper pass (no LLM calls) over the labeled set.
    # Reuses the same cached HybridSearcher (and its VectorStore) the graph just used above —
    # required when running against local-mode Qdrant, which only allows one open client per
    # storage path.
    searcher = get_hybrid_searcher()
    questions_by_id = {q.id: q.question for q in questions}
    retrieval_results = evaluate_retrieval_hit_rate(searcher, questions_by_id)
    retrieval_hit = hit_rate_score(retrieval_results)

    avg_latency = (
        sum(r.latency.total_ms for r in question_results) / len(question_results)
        if question_results
        else 0.0
    )
    cost = estimate_cost_usd(get_usage_log())

    report = EvalReport(
        question_results=question_results,
        global_claim_faithfulness=global_faithfulness,
        total_claims=total_claims,
        retrieval_hit_rate=retrieval_hit,
        retrieval_n=len(retrieval_results),
        total_cost_usd=cost,
        avg_latency_ms=avg_latency,
    )

    markdown = render_markdown(report)
    settings.eval_report_path.write_text(markdown)
    logger.info(
        "eval_complete",
        questions=len(question_results),
        faithfulness=global_faithfulness,
        retrieval_hit_rate=retrieval_hit,
        cost_usd=cost,
        report_path=str(settings.eval_report_path),
    )
    print(markdown)


if __name__ == "__main__":
    main()
