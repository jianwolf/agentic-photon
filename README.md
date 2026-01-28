# Agentic Photon

An intelligent news analysis pipeline powered by PydanticAI agents. Monitors RSS feeds from curated tech/AI sources, classifies stories by importance, and generates detailed analysis reports.

## LLM-as-a-Judge (Depth + Grounding)

This repo now includes an **LLM-as-a-judge** workflow to score report quality and expose real MLE issues:

- **Grounding compliance**: missing URLs, fake/placeholder sources, or citations that don't map to evidence.
- **Hallucinated specificity**: confident details not supported by sources or the prompt.
- **Depth vs cost trade-offs**: longer reports that add fluff instead of insight.
- **Reliability gaps**: failed calls (e.g., 502) that break evaluation pipelines.

**Workflow**

```bash
# 1) Generate flash vs pro reports
python main.py compare --limit 10

# 2) Judge depth + grounding
python eval/judge_reports.py --input-dir reports/compare/20260126_214722 \
  --model google-gla:gemini-3-pro-preview --max-concurrent 2
```

This produces `eval/judge_*.json` + `eval/judge_*.md` with per-story scores and winners.

## MLE Portfolio

Concrete ML engineering artifacts live in-repo so you can trace data -> evaluation -> monitoring:

- Data labeling workflow and schema: `data/README.md`, `data/label_schema.json`, and curated labels in `data/labels/`.
- Evaluation harnesses for classification + retrieval: `eval/README.md` (can write metrics JSON and optional per-query details).
- LLM-as-a-judge scoring for report depth + grounding: `eval/judge_reports.py` (batch scores + markdown summaries).
- Ablation tracking, model card, and latency benchmarks: `docs/ablation.md`, `docs/model_card.md`, `docs/benchmarks.md`.
- Monitoring snapshot tooling: `eval/monitoring_report.py` (important rate and top sources).

## Technical Highlights

This project demonstrates several engineering decisions optimized for **cost efficiency**, **consistency**, and **simplicity**:

| Challenge | Solution | Why This Approach |
|-----------|----------|-------------------|
| **API costs** | Local MLX classifier (3B model) | ~$0 for classification vs ~$0.01/story with cloud APIs |
| **Small model inconsistency** | Low temperature + Top-P + Chain-of-Thought | Research-backed techniques for deterministic output |
| **Retrieval quality** | Hybrid RAG (BM25 + Vector + RRF) | Combines keyword precision with semantic recall |
| **Vector search at scale** | Exact NN over SQLite | ANN complexity unjustified for <2K vectors |
| **Research depth vs cost** | Single Gemini call with grounding | 1 API call vs 5-15 with tool round-trips |
| **Deduplication** | SHA-256 hash of (normalized title + pub hour + source feed) | O(1) lookup, collision-resistant |
| **Evaluation rigor** | Label schema + seed/gold sets + eval harness | Reproducible metrics, ablations, and benchmarks |
| **Fail-safe classification** | Default to important on error | Never miss potentially important stories |

## Features

- **Agent-Based Architecture**: Separate classifier and researcher agents with structured outputs
- **Async Pipeline**: Concurrent feed fetching and processing with aiohttp
- **Local Classification**: Fast importance classification using local MLX model (Ministral-3B) on Apple Silicon
- **Deep Analysis**: Research agent with Gemini + Google Search grounding (no custom tool round-trips)
- **Flexible Output**: Markdown reports, webhooks, and JSONL alerts
- **Bilingual Support**: Chinese (zh) and English (en) with language-specific prompts
- **Optional Observability**: Logfire tracing (OpenTelemetry)
- **Evaluation Toolkit**: Label schema, classification/retrieval evals, ablations, model card, benchmarks

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export GEMINI_API_KEY=your-api-key
```

### 3. Run Pipeline

```bash
# Single run (default: Chinese output)
python main.py run

# English output
python main.py run --lang en

# Continuous monitoring (polls every 5 minutes)
python main.py run -c

# Custom poll interval
python main.py run -c --interval 600

# Generate a digest from existing markdown reports
python main.py digest --reports-dir reports
```

Note: `main.py run` starts a local MLX classifier by default (Apple Silicon). For non-Apple platforms, use the programmatic pipeline with `CLASSIFIER_MODEL` set to a remote model.

## MLE Workflows (Evaluation & Benchmarking)

```bash
# Build or refresh a labeling pool
python data/collect_label_pool.py --max-age-hours 168 --max-items 200 --out data/labels/pool.jsonl

# Extract a manual-only gold set from the seed labels
python data/extract_gold_labels.py --input data/labels/seed.jsonl --out data/labels/gold.jsonl

# Classification evaluation (remote model by default)
python eval/classification_eval.py --labels data/labels/seed.jsonl --out-metrics eval/metrics_seed.json

# Local MLX evaluation (Apple Silicon)
python eval/classification_eval.py --labels data/labels/gold.jsonl --mlx-model mlx-community/Ministral-3-3B-Instruct-2512

# Retrieval evaluation
python data/build_retrieval_queries.py --db-path news.db --out eval/queries.jsonl --query-mode title_summary
python eval/retrieval_eval.py --db-path news.db --queries eval/queries.jsonl --mode hybrid --out-details eval/retrieval_details.jsonl

# Benchmarks + monitoring snapshot
python eval/benchmark.py --labels data/labels/sample.jsonl --db-path news.db --out eval/benchmarks.json
python eval/classifier_benchmark.py --labels data/labels/seed.jsonl --max-items 50 --out eval/classifier_benchmark.json
python eval/monitoring_report.py --db-path news.db --days 7

# LLM-as-a-judge on report quality
python main.py compare --limit 10
python eval/judge_reports.py --input-dir reports/compare/20260126_214722 \
  --model google-gla:gemini-3-flash-preview --max-concurrent 2
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py run` | Execute pipeline once |
| `python main.py run -c` | Run continuously with polling |
| `python main.py run --lang en` | Output in English |
| `python main.py run --max-stories 5` | Limit to 5 most recent important stories |
| `python main.py run --classifier-model MODEL` | Use custom MLX model for classification |
| `python main.py status` | Show configuration and database stats |
| `python main.py recent --hours 48` | Display recent important stories |
| `python main.py analyze --title "..." --force` | Manually analyze a story |
| `python main.py digest --reports-dir reports` | Summarize existing reports into a digest |
| `python main.py compare --limit 10` | Compare flash vs pro researcher outputs |

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | **required** | Google Gemini API key |
| `LANGUAGE` | `zh` | Output language (`zh` or `en`) |
| `CLASSIFIER_MODEL` | `google-gla:gemini-3-flash-preview` | Classifier model for programmatic runs (CLI uses MLX by default) |
| `RESEARCHER_MODEL` | `google-gla:gemini-3-flash-preview` | Model for analysis |
| `RESEARCHER_MODEL_PRO` | `google-gla:gemini-3-pro-preview` | Alternate researcher model for comparisons |
| `MAX_AGE_HOURS` | `720` | Max story age (30 days) |
| `POLL_INTERVAL_SECONDS` | `300` | Polling interval (5 min) |
| `PRUNE_AFTER_DAYS` | `30` | Auto-delete older records |
| `MAX_WORKERS` | `8` | Concurrency for feed fetches and API calls |
| `MAX_RETRIES` | `3` | Retry attempts for API calls |
| `RETRY_BASE_DELAY` | `1.0` | Base delay for exponential backoff |
| `DB_PATH` | `news.db` | SQLite database path |
| `REPORTS_DIR` | `reports` | Markdown reports directory |
| `LOG_DIR` | `log` | Log file directory |
| `NOTIFICATION_WEBHOOK_URL` | | Optional webhook URL |
| `ALERTS_FILE` | | Optional JSONL alerts file |
| `ENABLE_LOGFIRE` | `false` | Enable Logfire tracing |
| `LOG_LEVEL` | `INFO` | Console log level |
| `LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `LOG_BACKUP_COUNT` | `30` | Rotated log files to keep |
| `LOG_MAX_BYTES` | `0` | Max log file size (0 = time-based rotation) |
| `LOGFIRE_TOKEN` | | Optional Logfire auth token |

## Architecture (v2 - Grounded)

```
RSS Feeds (17 sources)
       │
       ▼
  Async Fetch ──► Dedup ──► Classifier (Local MLX or remote LLM)
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
            Important                   Not Important
                 │                           │
                 ▼                           ▼
    ┌────────────┴────────────┐         Save & Skip
    │                         │
 URL Fetch              Hybrid RAG
 (deterministic)     (BM25 + Vector)
    │                         │
    └────────────┬────────────┘
                 │
                 ▼
    Researcher (Gemini + Google Search Grounding)
          Single API call per story
                 │
                 ▼
       Save + Embedding → Markdown/Webhook/JSONL
```

### Hybrid RAG Design

The system uses **hybrid retrieval** combining BM25 and vector search:

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| **BM25** | SQLite FTS5 | Keyword matching on title/summary |
| **Vector** | BGE-small-en-v1.5 (384-dim) | Semantic similarity |
| **Fusion** | Reciprocal Rank Fusion (RRF) | Combine rankings |

**Why RRF over other fusion methods?**

Reciprocal Rank Fusion was chosen over alternatives for specific reasons:

| Method | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Linear Combination** | Simple | Requires score normalization; BM25 and cosine scores aren't comparable | ❌ |
| **Learned Re-ranking** | Optimal weights | Needs training data; adds ML complexity | ❌ |
| **RRF** | Rank-based (no normalization); parameter-free | Slightly less optimal than learned | ✅ |

RRF formula: `score = Σ 1/(k + rank)` where k=60 (standard). It's robust, requires no tuning, and works well in practice.

**Why Exact NN (not ANN)?**

Vector search uses exact nearest neighbor (brute-force cosine similarity) rather than approximate nearest neighbor (HNSW, IVF, etc.):

- **Scale**: Low thousands of vectors with 30-day retention (configurable)
- **Speed**: Acceptable for local workloads; see `docs/benchmarks.md`
- **Accuracy**: 100% recall (no approximation loss)
- **Simplicity**: Zero dependencies (no FAISS, Annoy, sqlite-vec)

For this scale, exact NN is the right choice. ANN adds complexity without meaningful benefit until you have 10K+ vectors.

**Why BGE-small over larger models?**

| Model | Dimensions | Speed | Quality | Choice |
|-------|------------|-------|---------|--------|
| BGE-small-en-v1.5 | 384 | Fast on CPU (see `docs/benchmarks.md`) | Good | ✅ |
| BGE-base | 768 | Slower | Better | ❌ Overkill |
| OpenAI ada-002 | 1536 | API latency + cost | Best | ❌ Defeats local-first goal |

For news similarity (not fine-grained semantic search), BGE-small provides sufficient quality at 3x speed.

## Project Structure

```
agentic-photon/
├── agents/                 # PydanticAI agents
│   ├── classifier.py       # Fast importance classification
│   └── researcher.py       # Deep analysis with tools
├── models/                 # Pydantic data models
│   ├── story.py            # RSS story model
│   ├── classification.py   # Classification result
│   └── research.py         # Research report
├── tools/                  # Agent tools (legacy or supporting)
│   ├── search.py           # Legacy web search (not wired into pipeline)
│   ├── fetch.py            # Article fetcher
│   └── database.py         # History queries
├── data/                   # Labeling pipeline and schema
├── eval/                   # Evaluation and benchmarking scripts
├── docs/                   # Model card, ablations, benchmarks
├── observability/
│   ├── logging.py          # Enhanced logging with JSON/context
│   └── tracing.py          # Logfire integration
├── config.py               # Configuration management
├── database.py             # SQLite operations
├── feeds.py                # RSS fetching
├── mlx_server.py           # Local MLX model server manager
├── notifications.py        # Reports and alerts
├── pipeline.py             # Main orchestration
└── main.py                 # CLI entry point
```

## Local MLX Classification (Apple Silicon)

By default, `python main.py run` uses a local MLX model (Ministral-3B) for classification, which runs entirely on-device using Apple Silicon. This reduces API costs and latency. For non-Apple platforms, use the programmatic pipeline with `CLASSIFIER_MODEL` set to a remote model.

```bash
# Default: uses mlx-community/Ministral-3-3B-Instruct-2512
python main.py run

# Custom MLX model
python main.py run --classifier-model mlx-community/Qwen2.5-3B-Instruct

# Custom port for MLX server
python main.py run --mlx-port 8081
```

Requirements:
- macOS 15.0+ with Apple Silicon (M-series Chips)
- mlx-lm package: `pip install mlx-lm`

The first run will download the model (~2GB). The MLX server starts automatically and shuts down when the pipeline exits. If you want a remote classifier instead, run the `Pipeline` programmatically with `CLASSIFIER_MODEL` and skip the MLX CLI path.

### Consistency Optimizations for Small Models

A key challenge with small language models (3B parameters) is **classification inconsistency** — the same input can produce different outputs across runs due to sampling randomness. Through empirical testing, I identified three techniques that significantly improve determinism:

| Technique | Implementation | Impact |
|-----------|---------------|--------|
| **Low Temperature** | `temperature=0.1` | Reduces randomness in token selection; near-deterministic output |
| **Restrictive Nucleus Sampling** | `top_p=0.1` | Limits candidate tokens to highest-probability options |
| **Chain-of-Thought Prompting** | Structured reasoning steps | Forces systematic evaluation before final decision |

**Why these specific values?**

- **Temperature 0.1**: Near-greedy decoding. Note that `temperature=0` isn't truly deterministic in practice — GPU floating-point parallelism and model sharding introduce variability even with greedy sampling. OpenAI's docs acknowledge outputs are only "mostly deterministic" even with `seed` set.
- **Top-P 0.1 (not Top-K)**: Nucleus sampling is more reliable than Top-K for controlling output quality. A value of 0.1 restricts sampling to tokens within the top 10% probability mass, effectively limiting choices to high-confidence predictions.
- **Chain-of-Thought**: Instead of asking for a direct classification, the prompt requires:
  1. Identify the main topic and source type
  2. Check against "Important" criteria explicitly
  3. Check against "Not Important" criteria explicitly
  4. State final decision with reasoning

This structured approach reduces the model's tendency to make snap judgments based on surface patterns.

**Trade-off**: These optimizations prioritize consistency over creativity — appropriate for classification but not for generative tasks.

## Researcher Agent Design

The researcher agent uses **Gemini with Google Search grounding** instead of custom tool implementations. This is a deliberate architectural choice:

**Traditional Agentic Approach:**
```
User Query → LLM → Tool Call → Execute → LLM → Tool Call → ... → Final Answer
             ↑__________________________|
                    (5-15 round trips)
```

**Grounded Approach (this project):**
```
Context (article + RAG) → LLM with Search Grounding → Final Answer
                              (1 API call)
```

| Metric | Tool Round-trips | Grounded (Single Call) |
|--------|-----------------|------------------------|
| API calls per story | 5-15 | 1 |
| Latency | 10-30s | 3-8s |
| Token cost | High (repeated context) | Low |
| Failure modes | Many (tool errors, loops) | Few |

The trade-off: less flexibility in tool selection, but the researcher's job is analysis, not exploration. Pre-fetching context (article content + RAG results) before the LLM call provides sufficient grounding.

## Error Handling Philosophy

The pipeline follows a **fail-safe, continue-on-error** philosophy:

| Component | On Error | Rationale |
|-----------|----------|-----------|
| **Classifier** | Default to `is_important=True` | Never miss potentially important stories |
| **Article fetch** | Continue with empty content | RAG + grounding can compensate |
| **RAG search** | Continue with empty results | Grounding still works |
| **Researcher** | Return empty report, log error | Don't block other stories |
| **Webhook** | Log and continue | Don't lose local reports |
| **Feed fetch** | Skip feed, continue others | Partial results better than none |

This design prioritizes **availability over correctness** — appropriate for a monitoring system where missing data is worse than imperfect data.

## Optional Features

### Observability (Logfire)

Enable distributed tracing:

```bash
pip install logfire
export ENABLE_LOGFIRE=true
export LOGFIRE_TOKEN=your-token  # Optional
```

## Development

```bash
# Syntax check
python -m py_compile main.py pipeline.py

# Import test
python -c "from pipeline import Pipeline; print('OK')"

# Run with debug logging
python main.py run -v
```

## License

MIT
