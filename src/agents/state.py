"""Shared LangGraph state schema for the multi-agent financial RAG system.

Every node reads from and writes to this single `AgentState`. Fields that
accumulate across the verifier's retry loop use `operator.add` reducers so
re-running the filing retriever appends new evidence instead of overwriting it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class SubTask(TypedDict):
    description: str
    kind: Literal["numeric_lookup", "comparison", "qualitative", "multi_company"]
    companies: list[str]
    needs_filing: bool
    needs_market_data: bool


class RetrievedChunk(TypedDict):
    content: str
    company: str
    filing_type: str
    filing_date: str
    section: str
    score: float
    source: str  # "dense" | "bm25" | "hybrid"
    accession_number: str


class NumericClaim(TypedDict):
    text: str
    metric: str
    value: float
    unit: str
    company: str
    period: str
    source_span: str


class VerificationResult(TypedDict):
    claim: NumericClaim
    xbrl_value: float | None
    xbrl_tag: str | None
    status: Literal["confirmed", "contradicted", "unverifiable"]
    delta_pct: float | None


class TraceEvent(TypedDict):
    step: int
    agent: str
    duration_ms: float
    summary: str


class AgentState(TypedDict):
    # --- input ---
    query: str
    companies: list[str]

    # --- planner output ---
    subtasks: list[SubTask]

    # --- retrieval (accumulates across the verifier retry loop) ---
    filing_chunks: Annotated[list[RetrievedChunk], operator.add]
    market_news: list[dict]
    fundamentals: list[dict]

    # --- synthesis ---
    draft_answer: str
    numeric_claims: list[NumericClaim]
    citations: list[dict]

    # --- verification / control flow ---
    verification_results: list[VerificationResult]
    verification_passed: bool
    retry_count: int
    max_retries: int
    retry_hints: list[str]

    # --- output ---
    final_answer: str

    # --- observability ---
    trace: Annotated[list[TraceEvent], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(query: str, companies: list[str], max_retries: int = 2) -> AgentState:
    return AgentState(
        query=query,
        companies=companies,
        subtasks=[],
        filing_chunks=[],
        market_news=[],
        fundamentals=[],
        draft_answer="",
        numeric_claims=[],
        citations=[],
        verification_results=[],
        verification_passed=False,
        retry_count=0,
        max_retries=max_retries,
        retry_hints=[],
        final_answer="",
        trace=[],
        errors=[],
    )
