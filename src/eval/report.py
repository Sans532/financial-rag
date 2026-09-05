"""Renders the eval harness's collected metrics into a single markdown report."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.latency import LatencyBreakdown


@dataclass
class QuestionResult:
    id: str
    category: str
    difficulty: str
    question: str
    faithfulness_score: float | None  # None if the question had no numeric claims
    num_claims: int
    concept_coverage: float | None  # for qualitative questions
    latency: LatencyBreakdown
    errors: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    question_results: list[QuestionResult]
    global_claim_faithfulness: float  # confirmed / total, pooled across ALL claims
    total_claims: int
    retrieval_hit_rate: float
    retrieval_n: int
    total_cost_usd: float
    avg_latency_ms: float


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# Financial RAG — Evaluation Report\n")

    lines.append("## Headline metrics\n")
    lines.append(
        f"- **Faithfulness (numeric claim accuracy):** "
        f"{report.global_claim_faithfulness:.1%} ({report.total_claims} claims checked "
        f"against SEC XBRL ground truth)"
    )
    lines.append(
        f"- **Retrieval hit rate (hit@5):** {report.retrieval_hit_rate:.1%} "
        f"({report.retrieval_n} labeled questions)"
    )
    lines.append(f"- **Average latency:** {report.avg_latency_ms / 1000:.1f}s per query")
    lines.append(f"- **Total eval run cost (approx.):** ${report.total_cost_usd:.4f}\n")

    by_category: dict[str, list[QuestionResult]] = {}
    for r in report.question_results:
        by_category.setdefault(r.category, []).append(r)

    lines.append("## Results by category\n")
    lines.append("| Category | N | Avg. faithfulness | Avg. concept coverage |")
    lines.append("|---|---|---|---|")
    for category, results in sorted(by_category.items()):
        faith = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
        cov = [r.concept_coverage for r in results if r.concept_coverage is not None]
        avg_faith = sum(faith) / len(faith) if faith else None
        avg_cov = sum(cov) / len(cov) if cov else None
        lines.append(f"| {category} | {len(results)} | {_pct(avg_faith)} | {_pct(avg_cov)} |")
    lines.append("")

    lines.append("## Per-question detail\n")
    lines.append(
        "| ID | Category | Difficulty | Faithfulness | Claims | Concept coverage "
        "| Latency (s) | Errors |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in report.question_results:
        lines.append(
            f"| {r.id} | {r.category} | {r.difficulty} | {_pct(r.faithfulness_score)} | "
            f"{r.num_claims} | {_pct(r.concept_coverage)} | {r.latency.total_ms / 1000:.1f} | "
            f"{len(r.errors)} |"
        )
    lines.append("")

    lines.append("## Latency by agent (aggregated, ms)\n")
    agent_totals: dict[str, float] = {}
    for r in report.question_results:
        for agent, ms in r.latency.per_agent_ms.items():
            agent_totals[agent] = agent_totals.get(agent, 0.0) + ms
    lines.append("| Agent | Total ms | Avg ms/query |")
    lines.append("|---|---|---|")
    n = max(len(report.question_results), 1)
    for agent, total in sorted(agent_totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {agent} | {total:.0f} | {total / n:.0f} |")
    lines.append("")

    return "\n".join(lines)
