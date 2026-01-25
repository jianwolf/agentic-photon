# Local-First Agentic Roadmap

This document captures a concrete recommendation and detailed reasoning for a
future implementation that removes Gemini reliance and makes the system fully
agentic and free by default, while keeping model backends pluggable.

## Recommendation (TL;DR)
- Make local inference the default for both classifier and researcher.
- Introduce a tool-driven researcher agent that decides when to fetch, search,
  and retrieve context, with strict budgets and caching.
- Preserve the current "prefetch context" path as a non-agentic fallback that
  can be toggled for determinism and speed.
- Add evidence artifacts and citations to outputs to demonstrate MLE rigor.
- Keep the model interface pluggable so optional cloud models can be swapped
  without architectural changes.

## Why This Direction
1) Hiring manager signal
   - Demonstrates end-to-end system design, not just a wrapper around Gemini.
   - Shows understanding of tradeoffs: cost, latency, determinism, evals.
2) Local-first aligns with "free"
   - Classification already supports MLX; extend the same pattern to research.
   - Embeddings are already local (`embeddings.py`).
3) Agentic flow is more defensible
   - A tool-using agent that searches, verifies, and cites evidence is more
     impressive than a single grounded call.

## Design Goals
- Free-by-default: no paid API key required for the core pipeline.
- Deterministic control: budgets, caching, and fallbacks for reliability.
- Explainable outputs: evidence trail and citations.
- Minimal disruption: reuse existing code paths where possible.

## Proposed Architecture

### 1) Local model abstraction
Reuse the classifier pattern for the researcher to support local OpenAI-style
servers (MLX or other):
- Model string: `openai:{model_name}@http://127.0.0.1:PORT/v1`.
- Create a `_parse_local_model()` and `_create_model()` for researcher,
  paralleling `agents/classifier.py`.
- Make `GEMINI_API_KEY` optional and only required when the model string
  indicates Gemini.

Key files:
- `agents/researcher.py`
- `config.py`
- `main.py`

### 2) Agentic researcher with tools
Switch from Gemini built-in grounding to tool-based actions. The agent should
decide what to fetch and search rather than relying on a single grounded call.

Tools to expose (thin wrappers around existing utilities):
- `fetch_article(url)` -> `tools/fetch.py`.
- `search_web(query)` -> `tools/search.py` (add a free backend or a local
  search proxy).
- `query_related(query)` -> `Database.hybrid_search()` in `database.py`.

Implementation notes:
- Add tool budgets (max calls, total bytes, timeouts).
- Add caching in `tools/search.py` and `tools/fetch.py`.
- Preserve the current prefetch path as a fallback "non-agentic" mode:
  `config.researcher_mode = "agentic" | "prefetch"`.

Key files:
- `agents/researcher.py`
- `tools/search.py`
- `tools/fetch.py`
- `pipeline.py`
- `config.py`

### 3) Evidence pack + citations
Add evidence artifacts to the report so outputs can be audited.

Proposed additions:
- `models/research.py`: add `sources: list[EvidenceItem]` where EvidenceItem
  has `url`, `title`, `snippet`, `source_type`, `used_in`.
- `notifications.py`: include citations in markdown report output.
- `database.py`: optionally store evidence in a separate table or JSON column.

Benefits:
- Makes verification explicit.
- Enables future evaluation on citation coverage and source diversity.

### 4) Observability for agentic behavior
Extend `PipelineStats` to capture:
- tool call counts (search, fetch, rag).
- tool failures and average latency.
- evidence items produced per report.

Wire these into `observability/logging.py` for consistent reporting.

### 5) Evaluation plan (minimal, high signal)
Reuse existing eval utilities and add agentic quality checks:

Classification:
- Use `eval/classification_eval.py` and `eval/classifier_benchmark.py`.

Retrieval:
- Use `data/build_retrieval_queries.py` + `eval/retrieval_eval.py`.

Agentic analysis quality (new, small set):
- Create a rubric for 10-20 stories:
  - Factuality (1-5)
  - Evidence coverage (1-5)
  - Insight depth (1-5)
  - Citation correctness (1-5)
- Track in a simple JSONL file under `eval/`.

Runtime and cost:
- Measure `PipelineStats.duration` and tool timings.
- Compute cost: $0 for local models, or estimate per-token if a remote model
  is used.
- Record in `docs/benchmarks.md`.

## Minimal Implementation Plan

Phase 1: Local-first researcher and config cleanup
- Add local model parsing in `agents/researcher.py`.
- Make `GEMINI_API_KEY` optional in `config.py` and guard usage by model type.
- Add `RESEARCHER_MODE` config for "agentic" vs "prefetch".

Phase 2: Tool-based agentic research
- Introduce tool wrappers for fetch, search, and RAG.
- Add budgets and caching.
- Keep "prefetch" path for parity with current behavior.

Phase 3: Evidence pack and reporting
- Add `sources` to `models/research.py`.
- Update `notifications.py` to include citations.
- Add storage or log-only evidence if DB schema is not desired.

Phase 4: Evaluation and benchmarks
- Add an agentic rubric eval script under `eval/`.
- Update `docs/ablation.md`, `docs/benchmarks.md`, and `docs/model_card.md`.

## Key Tradeoffs and Risks
- Quality risk: local models may underperform Gemini in reasoning and fact
  checking; mitigate with tool budgets and deterministic prompts.
- Search availability: `tools/search.py` currently depends on paid APIs; add a
  free backend or self-hosted search to stay "free".
- Latency: local inference can be slower; use batching and caching.
- Determinism: agentic tools introduce nondeterminism; preserve prefetch mode.

## Open Questions
- Which free search backend is acceptable (DuckDuckGo HTML, SearxNG, or no web
  search with stronger RAG)?
- What is the minimum acceptable evidence coverage per report?
- Should evidence be persisted in SQLite or stored only in reports?

## Success Criteria
- No required paid API keys for default run.
- Comparable classification and retrieval metrics to current baselines.
- Evidence-backed reports with citation coverage above a defined threshold.
- Clear benchmarks and ablation results recorded in `docs/`.
