"""Per-query latency (by agent step) and approximate LLM token-cost tracking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.agents.state import TraceEvent

# $/1M tokens (input, output), paid standard tier. Update if the configured model changes.
# This project defaults to the free tier (see .env.example) — these numbers only matter if
# you outgrow the free daily quota and switch to billed usage. Flash-lite tiers priced as an
# approximation of the 2.5 flash-lite rate; verify against ai.google.dev/pricing before
# relying on this for anything beyond a rough order-of-magnitude estimate.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
}
DEFAULT_PRICING = (0.10, 0.40)


@dataclass
class LatencyBreakdown:
    per_agent_ms: dict[str, float]
    total_ms: float


def summarize_latency(trace: list[TraceEvent]) -> LatencyBreakdown:
    per_agent: dict[str, float] = defaultdict(float)
    for event in trace:
        per_agent[event["agent"]] += event["duration_ms"]
    return LatencyBreakdown(per_agent_ms=dict(per_agent), total_ms=sum(per_agent.values()))


def estimate_cost_usd(usage_log: list[dict]) -> float:
    """Approximate cost from logged Gemini usage records (see src/agents/llm.py)."""
    total = 0.0
    for entry in usage_log:
        input_price, output_price = MODEL_PRICING.get(entry["model"], DEFAULT_PRICING)
        total += entry["input_tokens"] / 1_000_000 * input_price
        total += entry["output_tokens"] / 1_000_000 * output_price
    return round(total, 6)
