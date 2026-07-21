"""Pydantic request/response models for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, examples=["How did Apple's gross margin change last quarter and why?"]
    )
    companies: list[str] = Field(
        default_factory=list, description="Tickers to scope the query to"
    )
    max_retries: int = Field(default=2, ge=0, le=5)


class CitationOut(BaseModel):
    company: str
    filing_type: str
    filing_date: str
    section: str
    accession_number: str


class VerificationOut(BaseModel):
    claim_text: str
    metric: str
    status: str
    xbrl_value: float | None
    delta_pct: float | None


class TraceStepOut(BaseModel):
    step: int
    agent: str
    duration_ms: float
    summary: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    verification: list[VerificationOut]
    faithfulness_score: float
    trace: list[TraceStepOut]
    errors: list[str]
    total_latency_ms: float
