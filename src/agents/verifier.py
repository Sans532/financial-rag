"""Verifier/Critic node — extracts numeric claims from the draft and checks them against
XBRL ground truth. Flags unsupported/contradicted claims and can route back to retrieval."""

from __future__ import annotations

import time

from src.agents.state import AgentState
from src.data.xbrl_client import XbrlClient
from src.eval.faithfulness import extract_numeric_claims, faithfulness_score, verify_claims
from src.logging_config import get_logger

logger = get_logger(__name__)

FAITHFULNESS_PASS_THRESHOLD = 0.8


def verifier_node(state: AgentState) -> dict:
    start = time.monotonic()

    if not state["draft_answer"]:
        return {
            "verification_passed": False,
            "verification_results": [],
            "trace": [
                {
                    "step": len(state["trace"]) + 1,
                    "agent": "verifier",
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "summary": "no draft to verify",
                }
            ],
        }

    claims = extract_numeric_claims(state["draft_answer"])
    with XbrlClient() as xbrl_client:
        results = verify_claims(claims, xbrl_client)

    score = faithfulness_score(results)
    passed = score >= FAITHFULNESS_PASS_THRESHOLD

    retry_hints: list[str] = []
    for r in results:
        if r["status"] == "contradicted":
            retry_hints.append(
                f"The claim \"{r['claim']['text']}\" does not match SEC XBRL data "
                f"(reported {r['xbrl_value']} for {r['claim']['metric']}/{r['claim']['company']}, "
                f"tag {r['xbrl_tag']}). Correct or remove this figure and re-cite from the filing."
            )
        elif r["status"] == "unverifiable":
            retry_hints.append(
                f"The claim \"{r['claim']['text']}\" ({r['claim']['metric']}/"
                f"{r['claim']['company']}) could not be checked against XBRL data. "
                "Ground it more precisely in the retrieved filing text, or qualify it as "
                "unverified."
            )

    return {
        "verification_results": results,
        "verification_passed": passed,
        "retry_hints": retry_hints,
        "trace": [
            {
                "step": len(state["trace"]) + 1,
                "agent": "verifier",
                "duration_ms": (time.monotonic() - start) * 1000,
                "summary": (
                    f"{len(claims)} claim(s) checked, faithfulness={score:.2f}, "
                    f"passed={passed}"
                ),
            }
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Terminal node: assembles the final answer, appending a verification footer that
    surfaces any claims that were contradicted or could not be checked — rather than
    silently dropping them."""
    start = time.monotonic()
    answer = state["draft_answer"] or "I was unable to generate an answer."

    flagged = [r for r in state["verification_results"] if r["status"] != "confirmed"]
    if flagged:
        footer_lines = ["\n\n---\n**Verification notes:**"]
        for r in flagged:
            if r["status"] == "contradicted":
                footer_lines.append(
                    f"- ⚠️ \"{r['claim']['text']}\" conflicts with SEC XBRL data "
                    f"(reported: {r['xbrl_value']})."
                )
            else:
                footer_lines.append(
                    f"- ℹ️ \"{r['claim']['text']}\" could not be independently verified."
                )
        answer += "\n".join(footer_lines)

    total = len(state["verification_results"])
    confirmed = total - len(flagged)
    return {
        "final_answer": answer,
        "trace": [
            {
                "step": len(state["trace"]) + 1,
                "agent": "finalize",
                "duration_ms": (time.monotonic() - start) * 1000,
                "summary": f"finalized answer ({confirmed}/{total} claims confirmed)",
            }
        ],
    }
