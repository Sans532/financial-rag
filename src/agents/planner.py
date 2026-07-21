"""Planner node — decomposes the user's question into typed sub-tasks."""

from __future__ import annotations

import time

from src.agents.llm import call_json
from src.agents.state import AgentState, SubTask
from src.data.universe import TICKER_TO_COMPANY
from src.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the planning agent in a financial-research system. Given a user's \
question about one or more public companies, break it into a short list of concrete sub-tasks \
so the retrieval agents know what to fetch.

Each sub-task must have:
- "description": a one-sentence statement of what's needed
- "kind": one of "numeric_lookup", "comparison", "qualitative", "multi_company"
- "companies": list of tickers involved (from the provided company list only)
- "needs_filing": true if answering requires SEC filing text (MD&A, risk factors, etc.)
- "needs_market_data": true if answering requires recent news or price/fundamentals data

Respond with ONLY a JSON array of sub-task objects. Keep it to 1-4 sub-tasks."""


def _fallback_subtasks(companies: list[str]) -> list[SubTask]:
    return [
        SubTask(
            description="Answer the user's question using filing text and market context.",
            kind="qualitative",
            companies=companies,
            needs_filing=True,
            needs_market_data=True,
        )
    ]


def planner_node(state: AgentState) -> dict:
    start = time.monotonic()
    known = [c for c in state["companies"] if c in TICKER_TO_COMPANY]
    user_prompt = (
        f"Question: {state['query']}\n"
        f"Companies mentioned (tickers): {known or state['companies']}"
    )

    try:
        raw = call_json(SYSTEM_PROMPT, user_prompt)
        subtasks: list[SubTask] = []
        for item in raw if isinstance(raw, list) else []:
            subtasks.append(
                SubTask(
                    description=str(item.get("description", "")),
                    kind=item.get("kind", "qualitative"),
                    companies=[c for c in item.get("companies", []) if c in TICKER_TO_COMPANY]
                    or known,
                    needs_filing=bool(item.get("needs_filing", True)),
                    needs_market_data=bool(item.get("needs_market_data", False)),
                )
            )
        if not subtasks:
            subtasks = _fallback_subtasks(known or state["companies"])
    except Exception as e:
        logger.error("planner_failed", exc_info=True)
        subtasks = _fallback_subtasks(known or state["companies"])
        return {
            "subtasks": subtasks,
            "errors": [f"planner: fell back to default sub-task ({e})"],
            "trace": [
                {
                    "step": len(state["trace"]) + 1,
                    "agent": "planner",
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "summary": "fallback plan used after planner error",
                }
            ],
        }

    return {
        "subtasks": subtasks,
        "trace": [
            {
                "step": len(state["trace"]) + 1,
                "agent": "planner",
                "duration_ms": (time.monotonic() - start) * 1000,
                "summary": f"produced {len(subtasks)} sub-task(s)",
            }
        ],
    }
