# Financial RAG — Multi-Agent SEC Filing Analysis

[github.com/Sans532/financial-rag](https://github.com/Sans532/financial-rag)

A production-grade, multi-agent RAG system that answers natural-language questions about
public companies by planning, retrieving SEC filings + market data, verifying every numeric
claim against structured XBRL ground truth, and synthesizing a cited answer.

Built as a portfolio project to demonstrate production-oriented AI engineering: an explicit,
inspectable agent state machine (LangGraph, not implicit chains), a real hybrid retrieval
stack over a persistent vector store, and — most importantly — an automated, quantitative
evaluation harness rather than ad-hoc testing.

## Why this exists

Most "RAG demo" projects stop at "it retrieves things and an LLM answers." The interesting
(and hard) part of shipping this kind of system is knowing whether the answers are actually
*correct*. This project treats that as the primary deliverable: every numeric claim the
system makes is programmatically cross-checked against SEC's machine-readable XBRL data,
and that check is both a live safety net (the Verifier agent can send the system back to
retrieval) and an offline metric (`python scripts/run_eval.py`).

---

## Architecture

```mermaid
graph TD
    START([User question]) --> Planner
    Planner -->|sub-tasks| FilingRetriever[Filing Retriever\nhybrid dense+BM25 + rerank]
    FilingRetriever --> MarketRetriever[Market/News Retriever\nyfinance + Finnhub]
    MarketRetriever --> Synthesizer[Synthesizer\ncited draft answer]
    Synthesizer --> Verifier[Verifier/Critic\nXBRL cross-check]
    Verifier -->|faithfulness >= 0.8, or retries left = 0| Finalize
    Verifier -->|faithfulness < 0.8 and retries remain| Retry[increment retry_count]
    Retry --> FilingRetriever
    Finalize --> END([Answer + citations + verification report])
```

**Agents** (`src/agents/`), each a LangGraph node reading/writing one shared `AgentState`:

| Agent | Responsibility |
|---|---|
| **Planner** | Decomposes the question into typed sub-tasks (numeric lookup, comparison, qualitative, multi-company), deciding which need filing text vs. market data. |
| **Filing Retriever** | Hybrid search (dense embeddings + BM25, fused via reciprocal rank fusion, cross-encoder reranked) over chunked SEC filings in Qdrant. |
| **Market/News Retriever** | Pulls recent news (Finnhub) and price/fundamentals (yfinance) for context. |
| **Synthesizer** | Drafts a cited answer using only the retrieved evidence. |
| **Verifier/Critic** | Extracts every numeric claim from the draft, cross-checks it against SEC XBRL structured data, and either passes the answer through or routes back to retrieval with specific hints about what to fix. |

The verifier's conditional edge is the one piece of real agentic control flow in the graph:
a failed verification with retry budget remaining re-enters `filing_retriever` (not the
planner — the sub-tasks are still valid, only the evidence needs to improve), carrying
`retry_hints` that target the retrieval query at the specific gap the verifier found.

**Shared state schema** (`src/agents/state.py`) — a single `TypedDict` passed through every
node, with `operator.add`-reduced fields (`filing_chunks`, `trace`, `errors`) so the retry
loop accumulates evidence instead of overwriting it.

---

## Data universe

18 large-cap tickers across 3 sectors, chosen for clean, consistently-tagged XBRL data and
reliable coverage across EDGAR, yfinance, and Finnhub's free tier:

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, NVDA, META, CRM |
| Finance | JPM, BAC, GS, MS, WFC, V |
| Healthcare | JNJ, UNH, PFE, MRK, ABBV, LLY |

Defined in `src/data/universe.py`.

## Zero-cost data sources

- **SEC EDGAR** (`data.sec.gov/submissions`, `www.sec.gov/Archives`) — 10-K/10-Q filing text. No key; rate-limited client-side to stay under SEC's fair-use cap, with a descriptive `User-Agent`.
- **SEC EDGAR XBRL** (`data.sec.gov/api/xbrl/`) — structured ground-truth financials (revenue, net income, EPS, etc.) used by the verifier and the eval harness's faithfulness metric.
- **yfinance** — price/fundamentals. No key.
- **Finnhub free tier** — company news. Requires a free email-signup API key (`FINNHUB_API_KEY`), no card.

No paid LLM-adjacent APIs are required for retrieval — the only billed API surface is the
Gemini model used for planning/synthesis/verification (`GOOGLE_API_KEY` + `GEMINI_MODEL`).
The default model is `gemini-3.1-flash-lite`, chosen for its free-tier daily request quota
(500 RPD vs. 20 RPD on `gemini-2.5-flash`) — each query fires several LLM calls (planning,
synthesis, claim extraction, and again on every verifier retry), so a full eval run needs
real quota headroom. Swap `GEMINI_MODEL` for a stronger model (e.g. `gemini-2.5-pro`) if you
want better qualitative/comparison answers and don't mind the lower free-tier quota or a
paid-tier cost.

---

## Retrieval design

- **Section-aware chunking** (`src/retrieval/chunking.py`): filings are split on `Item N`
  headers first, then any section too long for a single chunk is sub-split on paragraph
  boundaries with token overlap — not a naive fixed-size sliding window blind to filing
  structure. Item-number-to-title mapping is both filing-type- and Part-aware: 10-Ks and
  10-Qs use entirely different Item numbering (10-K Item 7 = MD&A; 10-Q Item 7 doesn't
  exist), and 10-Qs additionally reuse Item numbers 1-4 across Part I (Financial
  Statements, MD&A, ...) and Part II (Legal Proceedings, Risk Factors, ...) with different
  meanings each time — the chunker tracks the current Part while parsing so both
  occurrences resolve to the correct title.
- **Persistent vector store**: Qdrant (self-hosted via docker-compose), storing chunk text
  alongside company/filing-type/date/section metadata used for both retrieval filtering and
  citation.
- **Hybrid search**: dense (local `sentence-transformers` embeddings) + sparse (BM25) fused
  via reciprocal rank fusion, then reranked with a cross-encoder — not dense-only.
- **Citations**: every retrieved chunk carries its source metadata through synthesis into
  the final answer's citation list.

---

## Evaluation layer

This is the part of the project meant to be defensible in an interview, not an afterthought.

1. **Eval set** — `data/eval_set.json`, 30 hand-written questions spanning numeric lookup,
   QoQ/YoY comparison, qualitative ("why did X change"), and multi-company comparison, each
   with structured expected-answer characteristics (not exact strings): an XBRL metric +
   period + tolerance for numeric questions, or required concepts for qualitative ones.
2. **Faithfulness (headline metric)** — `src/eval/faithfulness.py`. Every numeric claim in
   a generated answer is extracted by the LLM into a structured `(metric, value, unit,
   company, period)` tuple, then cross-checked against the matching SEC XBRL fact. The
   headline number is `confirmed claims / total claims`, pooled across the whole eval run —
   this is the same code path used live by the Verifier agent, so "verified in eval" and
   "verified in production" are identical logic.
3. **Retrieval hit rate** — `src/eval/retrieval_hit_rate.py` + `data/retrieval_labels.json`,
   a small hand-labeled set of (question → acceptable filing-type/section) pairs, checked
   against what the hybrid retriever actually returns. It is deliberately named hit rate
   (recall@k), not precision: the denominator is *questions*, not retrieved chunks — what
   matters is whether answer-bearing evidence reached the synthesizer at all.
4. **Retrieval ablation** — `scripts/ablate_retrieval.py` quantifies what the hybrid stack
   actually buys over dense retrieval alone (see *Does hybrid retrieval earn its keep?*
   below). Pure retrieval, no LLM calls, so it runs free and needs no `GOOGLE_API_KEY`.
5. **Latency & cost** — `src/eval/latency.py` breaks down per-query latency by agent step
   (from the `trace` field every node writes to) and estimates LLM token cost from logged
   Gemini usage.
6. **Report** — regenerate everything with:

   ```bash
   python scripts/run_eval.py
   ```

   This writes `eval_report.md` (also servable via `GET /eval-report`) and prints it to
   stdout. Run this yourself with a configured `GOOGLE_API_KEY` and an ingested corpus —
   results depend on your ingested filings and will vary by model/config, so no fixed
   numbers are hardcoded here. Observed on a full 30-question run against all 18
   companies at the default `gemini-3.1-flash-lite` model: ~$0.05 (well within a paid
   tier's cost, and free-tier eligible), ~35-45 minutes wall-clock (client-side rate
   limiting paces requests under the free tier's per-minute cap — see Design notes below),
   and 46 numeric claims checked. The free tier's 500-requests/day cap means at most one
   or two full runs fit in a day before needing to wait for the next daily reset.


### Does hybrid retrieval earn its keep?

Asserting that hybrid + reranking beats dense retrieval is easy; measuring it is the
point. `scripts/ablate_retrieval.py` runs four arms over the same 30 labeled questions
with an identical candidate budget (`dense_k = sparse_k = 20`) and the same `final_k`, so
the only variable is how candidates are selected and ordered:

```bash
python scripts/ablate_retrieval.py --final-k 5
```

| Arm | hit@1 | hit@3 | hit@5 | MRR |
|---|---:|---:|---:|---:|
| Dense only | 33.3% | 46.7% | 56.7% | 0.418 |
| BM25 only | 30.0% | 53.3% | 56.7% | 0.412 |
| + RRF fusion | 36.7% | 53.3% | 63.3% | 0.462 |
| **+ cross-encoder rerank** | **40.0%** | **56.7%** | **70.0%** | **0.504** |

Dense-only to the full stack is **56.7% → 70.0% hit@5** (+23.5% relative), with MRR up
20.6%; fusion and reranking contributed +6.7pp each. The full-stack arm reproduces
`run_eval.py`'s own hit-rate number exactly, which is the check that the ablation harness
is measuring the same thing the eval does.

The per-category breakdown is the more useful result, because the two retrievers fail on
*different* question types:

| Category | Dense | BM25 | + RRF | + rerank |
|---|---:|---:|---:|---:|
| numeric_lookup | 75% | **88%** | 75% | 75% |
| comparison | 50% | 62% | **75%** | 62% |
| multi_company | 67% | 50% | **83%** | **83%** |
| qualitative | 38% | 25% | 25% | **62%** |

BM25 alone beats dense embeddings on numeric lookups — exact tickers and us-gaap metric
names are a lexical matching problem, not a semantic one. On open-ended "why did X change"
questions the reverse holds and BM25 actively hurts, dragging lexically-similar boilerplate
into the fused list; the cross-encoder is what recovers those (25% → 62%). That
complementarity is the argument for hybrid, and it is not visible in the headline number.

Measured per-stage cost, per company per query: dense 11ms, BM25 54ms, RRF <1ms,
cross-encoder 227ms — roughly 290ms for the full stack against 11ms for dense-only. That
is ~26x on the retrieval step but under 1% of end-to-end query latency, which is dominated
by LLM calls.

**Caveat, stated plainly:** n = 30. The 13.3pp headline gap is 5 questions fixed against 1
regression, and a single run with no seed variance. It is directional evidence, not a
significance claim. The ablation also measures retrieval quality only — whether better
retrieval yields more faithful *answers* would need a per-arm end-to-end re-run.

---

## Project structure

```
src/
  agents/       LangGraph state schema + each agent node + graph wiring
  retrieval/    chunking, embeddings, Qdrant vector store, BM25, hybrid search, reranker
  data/         SEC EDGAR / XBRL / yfinance / Finnhub clients, universe config, ingestion
  eval/         eval set loader, faithfulness metric, retrieval hit rate, latency, report
  api/          FastAPI app, routes, schemas
frontend/       Streamlit UI
tests/          pytest suite (chunking, retrieval, XBRL parsing, verifier, EDGAR client)
scripts/        ingest_universe.py, run_eval.py
data/           eval_set.json, retrieval_labels.json (checked in); raw/ (gitignored cache)
docker/         Dockerfile.api, Dockerfile.frontend
```

---

## Setup

### 1. Configure environment

```bash
cp .env.example .env
# then edit .env:
#   GOOGLE_API_KEY=...             (required — free key at aistudio.google.com/apikey)
#   FINNHUB_API_KEY=...            (optional — free signup at finnhub.io; news is skipped without it)
#   SEC_USER_AGENT="your project you@email.com"   (required by SEC — identify yourself)
```

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

This starts Qdrant (`localhost:6333`), the API (`localhost:8000`), and the Streamlit UI
(`localhost:8501`). No Docker Desktop on macOS? [Colima](https://github.com/abiosoft/colima)
is a fully CLI-installable/scriptable alternative (`brew install colima docker
docker-compose && colima start`) — no GUI app or manual permission dialogs required.

Then, one-off, to populate the vector store (only needed once, or when you want more
companies than whatever's already ingested):

```bash
source .venv/bin/activate   # host-side venv, see step 3 — the ingest script talks to
                             # the dockerized Qdrant over QDRANT_URL=http://localhost:6333
python scripts/ingest_universe.py --filings-per-form 1 --tickers AAPL   # or omit --tickers for all 18
```

(You can also run it inside the container: `docker compose exec api python
scripts/ingest_universe.py ...` — `scripts/` and `data/` are both bind-mounted into the
`api` service, so this works without rebuilding the image.)

### 3. Or run without Docker at all

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Qdrant can run in two ways:

- **Server mode** (matches docker-compose): point `QDRANT_URL` at a running instance, e.g.
  `docker run -p 6333:6333 qdrant/qdrant:v1.12.4` if you have Docker but not Compose set up.
- **Embedded/local mode** (no Docker at all): set `QDRANT_LOCAL_PATH=data/qdrant_local` in
  `.env` (leave `QDRANT_URL` alone — local mode takes priority when both are set). This runs
  Qdrant on-disk in-process — a real persistent store, just no server. The one constraint:
  only one process can hold that path open at a time, so don't run the API and
  `ingest_universe.py`/`run_eval.py` concurrently against the same local path.

```bash
# Ingest filings for the company universe (downloads from EDGAR, chunks, embeds, upserts)
python scripts/ingest_universe.py --filings-per-form 1

# Start the API
uvicorn src.api.main:app --reload

# In another terminal, start the UI
streamlit run frontend/app.py
```

### 4. Run the eval suite

```bash
python scripts/run_eval.py            # full 30-question set
python scripts/run_eval.py --limit 5  # quick smoke run
```

### 5. Run tests

```bash
pytest tests/ -v
ruff check .
mypy src/   # advisory (non-blocking) in CI — see .github/workflows/ci.yml. Kept advisory
            # because numpy's bundled stubs use syntax mypy can choke on depending on the
            # interpreter version; it isn't a signal on this project's own code quality.
```

---

## Example queries

- "How did Apple's gross margin change last quarter and why?"
- "Compare JPMorgan Chase and Bank of America's net income year-over-year."
- "What risk factors did NVIDIA highlight in its most recent 10-K?"
- "What was Pfizer's R&D expense in its latest 10-Q?"

## API

```
POST /query
{
  "question": "How did Apple's gross margin change last quarter and why?",
  "companies": ["AAPL"],
  "max_retries": 2
}
```

Returns the final answer, citations, per-claim verification results, faithfulness score,
full agent trace (per-step latency), and any non-fatal errors encountered.

```
GET /eval-report   # markdown from the last `python scripts/run_eval.py` run
GET /health
```

---

## Design notes / known limitations

- The BM25 index is rebuilt from Qdrant's stored payloads on every hybrid search call rather
  than persisted separately. At this project's scale (a handful of filings per company) this
  is fast enough (~ms) and avoids keeping two indexes in sync; it would need to change for a
  much larger corpus.
- Numeric-claim → XBRL period matching (`src/eval/faithfulness.py::_match_period`) is a
  best-effort regex match on quarter/year mentions in the model's stated period. If a
  specific period is named but no XBRL fact matches it, the claim is reported
  **unverifiable** rather than compared against a different period's fact, avoiding false
  "contradicted" verdicts from a period mismatch on our side rather than a real error in
  the claim. A more subtle version of this was found via live full-eval runs: a single
  filing's balance sheet reports the current period alongside comparative prior periods
  (prior year-end, prior-year same-quarter), and XBRL's `fiscal_year`/`fiscal_period`
  label describes the *filing's* reporting context, not which period a given fact covers —
  so multiple facts can share an identical `(fiscal_year, fiscal_period, filed)` tuple
  (confirmed directly against JPMorgan's real XBRL data: three facts, all labeled
  "Q1 2026", covering three different period-end dates). Fixed by disambiguating on
  latest `period_end` rather than an arbitrary tie-break — this alone moved the full
  30-question eval's faithfulness score from ~5% to ~48%, i.e. it was the dominant driver
  of low scores, not model hallucination. Comparison tolerance is intentionally generous
  (`CONFIRMED_TOLERANCE_PCT = 2.0`) to absorb rounding without masking real errors.
- The XBRL metric taxonomy (`src/data/xbrl_client.py::METRIC_TO_TAGS`) initially conflated
  ongoing R&D expense with one-time "acquired in-process R&D" charges (common in pharma
  filings after an acquisition) under a single metric — comparing a $9B one-time charge
  against a $3.6B ongoing-R&D XBRL fact always "contradicted" even when both figures were
  individually correct. Split into separate `research_and_development` /
  `acquired_iprd_expense` metrics, with the claim-extraction prompt explicitly instructed
  to distinguish them. Also found Pfizer reports zero facts under the standard
  `ResearchAndDevelopmentExpense` tag (it uses
  `ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost` instead) — added as a
  fallback alias. This taxonomy is necessarily incomplete; other companies may use other
  nonstandard tags not yet covered.
- The claim-extraction prompt originally pulled any dollar figure resembling a metric
  name, including narrative cost/revenue "drivers" mentioned in MD&A prose (e.g. "$180
  million in higher spending on X") — these have no standalone XBRL fact to check them
  against and always read as unverifiable/contradicted. Tightened to only extract
  claims restating a period's aggregate reported figure.
- 10-K vs. 10-Q Item numbering is handled explicitly (see Retrieval design above) — this
  was a real bug found via live end-to-end testing, not just synthetic unit tests: every
  ingested 10-Q chunk was silently mislabeled with 10-K section titles until fixed, which
  had zeroed out the retrieval hit-rate metric for any 10-Q-sourced question.
- Rate limiting for EDGAR is a simple client-side token bucket (`src/data/edgar_client.py`)
  tuned conservatively under SEC's 10 req/sec fair-use cap. Gemini calls (`src/agents/
  llm.py`) are similarly paced client-side and retry 429s using the API's own suggested
  cooldown — required in practice, since this pipeline's per-question call volume
  (planner + synthesizer + claim extraction, doubling on every verifier retry) exceeds
  the free tier's per-minute cap easily; the free tier's 500-requests/day cap is a harder
  ceiling this doesn't work around and will still exhaust across repeated full eval runs
  in a single day.
