"""LangGraph wiring for the multi-agent financial RAG system.

Graph shape:

    START -> planner -> filing_retriever -> market_retriever -> synthesizer -> verifier
                              ^                                                   |
                              |________________ (verification failed, retries left)
                                                                                   |
                                                                    (passed, or retries exhausted)
                                                                                   v
                                                                              finalize -> END

The verifier's conditional edge is the one piece of "agentic" control flow: it decides,
based on the faithfulness check, whether to loop back to retrieval with hints about what
was wrong, or to terminate.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agents.filing_retriever import filing_retriever_node
from src.agents.market_retriever import market_retriever_node
from src.agents.planner import planner_node
from src.agents.state import AgentState
from src.agents.synthesizer import synthesizer_node
from src.agents.verifier import finalize_node, verifier_node


def route_after_verification(state: AgentState) -> str:
    if state["verification_passed"]:
        return "finalize"
    if state["retry_count"] >= state["max_retries"]:
        return "finalize"
    return "retry"


def _increment_retry(state: AgentState) -> dict:
    return {"retry_count": state["retry_count"] + 1}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("filing_retriever", filing_retriever_node)
    graph.add_node("market_retriever", market_retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "filing_retriever")
    graph.add_edge("filing_retriever", "market_retriever")
    graph.add_edge("market_retriever", "synthesizer")
    graph.add_edge("synthesizer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {"retry": "increment_retry", "finalize": "finalize"},
    )
    # Retry loop re-enters retrieval (not the planner) — the sub-tasks are still valid,
    # only the evidence needs to improve.
    graph.add_edge("increment_retry", "filing_retriever")
    graph.add_edge("finalize", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
