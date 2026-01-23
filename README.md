# Agentic Photon

An intelligent news analysis pipeline powered by PydanticAI agents. Monitors RSS feeds from curated tech/AI sources, classifies stories by importance, and generates detailed analysis reports.

## Features

- **Agent-Based Architecture**: Separate classifier and researcher agents with structured outputs
- **Async Pipeline**: Concurrent feed fetching and processing with aiohttp
- **Local Classification**: Fast importance classification using local MLX model (Ministral-3B) on Apple Silicon
- **Deep Analysis**: Research agent with tools for web search, article fetching, and history lookup
- **Flexible Output**: Markdown reports, webhooks, and JSONL alerts
- **Bilingual Support**: Chinese (zh) and English (en) output
- **Optional Enhancements**: Vector memory (ChromaDB) and observability (Logfire)

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

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | **required** | Google Gemini API key |
| `LANGUAGE` | `zh` | Output language (`zh` or `en`) |
| `CLASSIFIER_MODEL` | `google-gla:gemini-3-flash-preview` | Model for classification (overridden by `--classifier-model` CLI arg) |
| `RESEARCHER_MODEL` | `google-gla:gemini-3-flash-preview` | Model for analysis |
| `MAX_AGE_HOURS` | `720` | Max story age (30 days) |
| `POLL_INTERVAL_SECONDS` | `300` | Polling interval (5 min) |
| `DB_PATH` | `news.db` | SQLite database path |
| `REPORTS_DIR` | `reports` | Markdown reports directory |
| `NOTIFICATION_WEBHOOK_URL` | | Optional webhook URL |
| `ALERTS_FILE` | | Optional JSONL alerts file |
| `ENABLE_MEMORY` | `false` | Enable ChromaDB vector store |
| `ENABLE_LOGFIRE` | `false` | Enable Logfire tracing |
| `GOOGLE_API_KEY` | | Google Custom Search API key (enables web search) |
| `GOOGLE_CSE_ID` | | Google Custom Search Engine ID |
| `SERPAPI_KEY` | | SerpAPI key (alternative search backend) |

## Architecture (v2 - Grounded)

```
RSS Feeds (17 sources)
       │
       ▼
  Async Fetch ──► Dedup ──► Classifier (Local LLM)
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

**Why Exact NN (not ANN)?**

Vector search uses exact nearest neighbor (brute-force cosine similarity) rather than approximate nearest neighbor (HNSW, IVF, etc.):

- **Scale**: ~300-1500 vectors (30 days × 10-50 important stories/day)
- **Speed**: <1ms for 1000 vectors — fast enough
- **Accuracy**: 100% recall (no approximation loss)
- **Simplicity**: Zero dependencies (no FAISS, Annoy, sqlite-vec)

For this scale, exact NN is the right choice. ANN adds complexity without meaningful benefit until you have 10K+ vectors.

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
├── tools/                  # Agent tools
│   ├── search.py           # Web search (Google CSE or SerpAPI)
│   ├── fetch.py            # Article fetcher
│   └── database.py         # History queries
├── memory/                 # Optional features
│   └── vector_store.py     # ChromaDB semantic search
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

By default, the pipeline uses a local MLX model (Ministral-3B) for classification, which runs entirely on-device using Apple Silicon. This reduces API costs and latency.

```bash
# Default: uses Ministral-3B-Instruct
python main.py run

# Custom MLX model
python main.py run --classifier-model mlx-community/Qwen2.5-3B-Instruct

# Custom port for MLX server
python main.py run --mlx-port 8081
```

Requirements:
- macOS 15.0+ with Apple Silicon (M1/M2/M3)
- mlx-lm package: `pip install mlx-lm`

The first run will download the model (~2GB). The MLX server starts automatically and shuts down when the pipeline exits.

## Optional Features

### Vector Memory (ChromaDB)

Enable semantic search across story history:

```bash
pip install chromadb
export ENABLE_MEMORY=true
```

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
