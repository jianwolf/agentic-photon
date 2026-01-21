# Agentic Photon

An intelligent news analysis pipeline powered by PydanticAI agents. Monitors RSS feeds from curated tech/AI sources, classifies stories by importance, and generates detailed analysis reports.

## Features

- **Agent-Based Architecture**: Separate classifier and researcher agents with structured outputs
- **Async Pipeline**: Concurrent feed fetching and processing with aiohttp
- **Smart Classification**: Fast importance classification using Gemini Flash
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
| `python main.py status` | Show configuration and database stats |
| `python main.py recent --hours 48` | Display recent important stories |
| `python main.py analyze --title "..." --force` | Manually analyze a story |

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | **required** | Google Gemini API key |
| `LANGUAGE` | `zh` | Output language (`zh` or `en`) |
| `CLASSIFIER_MODEL` | `google-gla:gemini-2.0-flash` | Model for classification |
| `RESEARCHER_MODEL` | `google-gla:gemini-2.0-flash` | Model for analysis |
| `MAX_AGE_HOURS` | `720` | Max story age (30 days) |
| `POLL_INTERVAL_SECONDS` | `300` | Polling interval (5 min) |
| `DB_PATH` | `news.db` | SQLite database path |
| `REPORTS_DIR` | `reports` | Markdown reports directory |
| `NOTIFICATION_WEBHOOK_URL` | | Optional webhook URL |
| `ALERTS_FILE` | | Optional JSONL alerts file |
| `ENABLE_MEMORY` | `false` | Enable ChromaDB vector store |
| `ENABLE_LOGFIRE` | `false` | Enable Logfire tracing |

## Architecture

```
RSS Feeds (17 sources)
       │
       ▼
  Async Fetch ──► Dedup ──► Classifier Agent
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
            Important                   Not Important
                 │                           │
                 ▼                           ▼
        Researcher Agent              Save & Skip
                 │
    ┌────────────┼────────────┐
    │            │            │
 Search     Fetch URL    Query DB
    │            │            │
    └────────────┼────────────┘
                 │
                 ▼
          Research Report
                 │
    ┌────────────┼────────────┐
    │            │            │
 Markdown    Webhook      JSONL
```

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
│   ├── search.py           # Web search (placeholder)
│   ├── fetch.py            # Article fetcher
│   └── database.py         # History queries
├── memory/                 # Optional features
│   └── vector_store.py     # ChromaDB semantic search
├── observability/
│   └── tracing.py          # Logfire integration
├── config.py               # Configuration management
├── database.py             # SQLite operations
├── feeds.py                # RSS fetching
├── notifications.py        # Reports and alerts
├── pipeline.py             # Main orchestration
└── main.py                 # CLI entry point
```

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
