"""Synthesizer node — drafts the answer with inline citations back to filings/news."""

from __future__ import annotations

import time

from src.agents.llm import call_text
from src.agents.state import AgentState

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer the user's question using ONLY \
the provided filing excerpts and market data. Every numeric claim (dollar amounts, percentages, \
ratios) MUST be traceable to one of the provided sources.

Cite sources inline using the format [Company, FilingType, FilingDate, Section] for filing \
excerpts, or [Company, News, Date] for news items. If the provided context is insufficient to \
fully answer, say so explicitly rather than guessing.

Be concise. Lead with the direct answer, then supporting detail."""


def _format_context(state: AgentState) -> str:
    parts: list[str] = []

    if state["filing_chunks"]:
        parts.append("## SEC Filing Excerpts")
        for c in state["filing_chunks"]:
            parts.append(
                f"[{c['company']}, {c['filing_type']}, {c['filing_date']}, {c['section']}]\n"
                f"{c['content'][:1500]}"
            )

    if state["fundamentals"]:
        parts.append("## Market Fundamentals")
        for f in state["fundamentals"]:
            parts.append(str(f))

    if state["market_news"]:
        parts.append("## Recent News")
        for n in state["market_news"]:
            parts.append(f"[{n['ticker']}, News, {n['datetime']}] {n['headline']}: {n['summary']}")

    if state.get("retry_hints"):
        parts.append("## Verifier feedback from previous draft (address these)")
        parts.extend(state["retry_hints"])

    return "\n\n".join(parts) if parts else "No context retrieved."


def synthesizer_node(state: AgentState) -> dict:
    start = time.monotonic()
    context = _format_context(state)
    user_prompt = f"Question: {state['query']}\n\nContext:\n{context}"

    try:
        draft = call_text(SYSTEM_PROMPT, user_prompt, max_tokens=1500)
    except Exception as e:
        return {
            "draft_answer": "",
            "errors": [f"synthesizer: generation failed ({e})"],
            "trace": [
                {
                    "step": len(state["trace"]) + 1,
                    "agent": "synthesizer",
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "summary": "generation failed",
                }
            ],
        }

    citations = [
        {
            "company": c["company"],
            "filing_type": c["filing_type"],
            "filing_date": c["filing_date"],
            "section": c["section"],
            "accession_number": c["accession_number"],
        }
        for c in state["filing_chunks"]
    ]

    return {
        "draft_answer": draft,
        "citations": citations,
        "trace": [
            {
                "step": len(state["trace"]) + 1,
                "agent": "synthesizer",
                "duration_ms": (time.monotonic() - start) * 1000,
                "summary": f"drafted answer ({len(draft)} chars, {len(citations)} citation(s))",
            }
        ],
    }
