# Agentic Photon

An intelligent news analysis pipeline powered by PydanticAI agents. Monitors RSS feeds from curated tech/AI sources, classifies stories by importance, and generates detailed analysis reports.

## Features

- **Agent-Based Architecture**: Separate classifier and researcher agents with structured outputs
- **Async Pipeline**: Concurrent feed fetching and processing with aiohttp
- **Smart Classification**: Fast importance classification using Gemini Flash
- **Deep Analysis**: Research agent with tools for web search, article fetching, and history lookup
- **Flexible Output**: Markdown reports, webhooks, and JSONL alerts
- **Optional Enhancements**: Vector memory (ChromaDB) and observability (Logfire)

## Installation

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-api-key
```

## Quick Start

```bash
# Run once
python main.py run

# English output
python main.py run --lang en

# Continuous monitoring
python main.py run -c

# Check status
python main.py status

# Recent stories
python main.py recent --hours 48

# Analyze specific story
python main.py analyze --title "AI Breakthrough" --force
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | required | Google Gemini API key |
| `LANGUAGE` | `zh` | Output language (`zh`/`en`) |
| `CLASSIFIER_MODEL` | `google-gla:gemini-2.0-flash` | Classification model |
| `RESEARCHER_MODEL` | `google-gla:gemini-2.0-flash` | Research model |
| `MAX_AGE_HOURS` | `720` | Max story age (30 days) |
| `POLL_INTERVAL_SECONDS` | `300` | Polling interval |
| `DB_PATH` | `news.db` | Database path |
| `NOTIFICATION_WEBHOOK_URL` | | Webhook URL |
| `ALERTS_FILE` | | JSONL alerts file |
| `ENABLE_MEMORY` | `false` | ChromaDB vector store |
| `ENABLE_LOGFIRE` | `false` | Logfire tracing |

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
├── agents/
│   ├── classifier.py    # ClassifierAgent
│   └── researcher.py    # ResearcherAgent with tools
├── models/
│   ├── story.py         # Story model
│   ├── classification.py # ClassificationResult
│   └── research.py      # ResearchReport, Analysis
├── tools/
│   ├── search.py        # Web search (placeholder)
│   ├── fetch.py         # Article fetcher
│   └── database.py      # History queries
├── memory/
│   └── vector_store.py  # ChromaDB (optional)
├── observability/
│   └── tracing.py       # Logfire (optional)
├── config.py            # Configuration
├── database.py          # SQLite operations
├── feeds.py             # RSS fetching
├── notifications.py     # Alerts
├── pipeline.py          # Orchestration
└── main.py              # CLI
```

## CLI Commands

```bash
python main.py run              # Single run
python main.py run -c           # Continuous
python main.py run --lang en    # English output
python main.py status           # Show config/stats
python main.py recent           # Recent important stories
python main.py analyze --title "..." --force
```

## License

MIT
