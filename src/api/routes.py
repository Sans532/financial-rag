"""FastAPI routes."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from src.agents.graph import get_graph
from src.agents.state import initial_state
from src.api.schemas import (
    CitationOut,
    QueryRequest,
    QueryResponse,
    TraceStepOut,
    VerificationOut,
)
from src.config import get_settings
from src.eval.faithfulness import faithfulness_score
from src.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.monotonic()
    graph = get_graph()
    state = initial_state(request.question, request.companies, max_retries=request.max_retries)

    try:
        result = graph.invoke(state)
    except Exception:
        logger.error("query_graph_invoke_failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Agent pipeline failed") from None

    score = faithfulness_score(result["verification_results"])

    return QueryResponse(
        answer=result["final_answer"] or result["draft_answer"],
        citations=[CitationOut(**c) for c in result["citations"]],
        verification=[
            VerificationOut(
                claim_text=v["claim"]["text"],
                metric=v["claim"]["metric"],
                status=v["status"],
                xbrl_value=v["xbrl_value"],
                delta_pct=v["delta_pct"],
            )
            for v in result["verification_results"]
        ],
        faithfulness_score=score,
        trace=[TraceStepOut(**t) for t in result["trace"]],
        errors=result["errors"],
        total_latency_ms=(time.monotonic() - start) * 1000,
    )


@router.get("/eval-report")
def eval_report() -> dict:
    settings = get_settings()
    if not settings.eval_report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No eval report found. Run `python scripts/run_eval.py` first.",
        )
    return {"report_markdown": settings.eval_report_path.read_text()}


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}
