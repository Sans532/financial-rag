"""Loader for the structured eval question set (data/eval_set.json)."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from src.config import get_settings

Category = Literal["numeric_lookup", "comparison", "qualitative", "multi_company"]


class ExpectedNumeric(BaseModel):
    type: Literal["numeric"] = "numeric"
    metric: str
    period: str  # "latest_10Q" | "latest_10K" | explicit e.g. "FY2023"
    tolerance_pct: float = 5.0


class ExpectedQualitative(BaseModel):
    type: Literal["qualitative"] = "qualitative"
    required_concepts: list[str]


class ExpectedSources(BaseModel):
    filing_type: str
    section: str


class EvalQuestion(BaseModel):
    id: str
    category: Category
    question: str
    companies: list[str]
    expected: Annotated[ExpectedNumeric | ExpectedQualitative, Field(discriminator="type")]
    expected_sources: ExpectedSources
    difficulty: Literal["easy", "medium", "hard"] = "medium"


def load_eval_set() -> list[EvalQuestion]:
    settings = get_settings()
    with open(settings.eval_set_path) as f:
        raw = json.load(f)
    return [EvalQuestion.model_validate(q) for q in raw]
