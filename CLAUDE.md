# Agentic Photon - Developer Guide

## Overview

Agentic Photon is a PydanticAI-powered news analysis pipeline that:
1. Fetches RSS feeds from curated tech/AI sources
2. Classifies stories by importance using a fast classifier agent
3. Performs deep analysis on important stories using a researcher agent
4. Generates markdown reports and sends notifications

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

## Key Files Reference

### Core Pipeline

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | CLI entry point | `main()`, `cmd_run()`, `cmd_status()` |
| `pipeline.py` | Main orchestration | `Pipeline.run_once()`, `Pipeline.run_continuous()` |
| `config.py` | Configuration | `Config.load()`, `Config.validate()` |
| `database.py` | SQLite storage | `Database.save()`, `Database.seen_hashes()` |
| `feeds.py` | RSS fetching | `fetch_all_feeds()` |
| `notifications.py` | Output | `save_markdown_report()`, `notify_batch()` |

### Agents

| File | Class | Description |
|------|-------|-------------|
| `agents/classifier.py` | `ClassifierAgent` | Fast importance classification using Gemini Flash |
| `agents/researcher.py` | `ResearcherAgent` | Deep analysis with web search, fetch, and history tools |

### Models

| File | Classes | Description |
|------|---------|-------------|
| `models/story.py` | `Story` | RSS feed item with dedup hash |
| `models/classification.py` | `ClassificationResult`, `ImportanceCategory` | Classification output |
| `models/research.py` | `ResearchReport`, `Analysis` | Research output |

### Tools (Used by ResearcherAgent)

| File | Function | Status |
|------|----------|--------|
| `tools/search.py` | `web_search()` | Placeholder - returns empty results |
| `tools/fetch.py` | `fetch_article()` | Active - fetches and parses HTML |
| `tools/database.py` | `query_related_stories()` | Active - keyword search on history |

### Optional Features

| File | Feature | Enable With |
|------|---------|-------------|
| `memory/vector_store.py` | Semantic search via ChromaDB | `ENABLE_MEMORY=true` |
| `observability/tracing.py` | Distributed tracing via Logfire | `ENABLE_LOGFIRE=true` |
| `observability/logging.py` | Enhanced logging with JSON/context | `LOG_FORMAT=json` |

## Pipeline Flow Details

1. **Prune**: Delete records older than `PRUNE_AFTER_DAYS`
2. **Fetch**: Concurrent RSS fetching with SSL fallback
3. **Dedup**: Filter stories already in database by hash
4. **Classify**: Run `ClassifierAgent.classify_batch()` on new stories
5. **Split**: Separate important (researcher) from not-important (skip)
6. **Analyze**: Run `ResearcherAgent.analyze_batch()` on important stories
7. **Save**: Store all results in SQLite
8. **Notify**: Generate reports, send webhooks, append JSONL

## Configuration Reference

```bash
# === Required ===
GEMINI_API_KEY=your-api-key

# === Models (PydanticAI format) ===
CLASSIFIER_MODEL=google-gla:gemini-3-flash-preview
RESEARCHER_MODEL=google-gla:gemini-3-flash-preview

# === Output ===
LANGUAGE=zh                    # 'zh' (Chinese) or 'en' (English)
DB_PATH=news.db               # SQLite database
REPORTS_DIR=reports           # Markdown reports
LOG_DIR=log                   # Log files

# === Pipeline Behavior ===
MAX_AGE_HOURS=720             # 30 days
POLL_INTERVAL_SECONDS=300     # 5 minutes
MAX_WORKERS=8                 # Concurrent operations

# === Notifications ===
NOTIFICATION_WEBHOOK_URL=     # Optional webhook
ALERTS_FILE=                  # Optional JSONL file

# === Optional Features ===
ENABLE_MEMORY=false           # ChromaDB vector search
ENABLE_LOGFIRE=false          # Distributed tracing

# === Logging ===
LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=text               # 'text' or 'json' for structured logging
LOG_BACKUP_COUNT=30           # Number of rotated log files to keep
LOG_MAX_BYTES=0               # Max file size (0 = time-based rotation)
```

## CLI Commands

```bash
# Run once
python main.py run

# Run with English output
python main.py run --lang en

# Continuous mode with custom interval
python main.py run -c --interval 600

# Check configuration
python main.py status

# View recent stories
python main.py recent --hours 48

# Manually analyze a story
python main.py analyze --title "Story Title" --force

# Debug mode
python main.py run -v
```

## Data Models

### Story
```python
class Story(BaseModel):
    title: str                    # Article headline
    description: str              # HTML/text from RSS
    pub_date: datetime            # Publication timestamp (UTC)
    source_url: str               # RSS feed URL

    @property
    def hash(self) -> str:        # 16-char SHA-256 dedup key
    @property
    def publishers(self) -> list[str]:  # Extracted from HTML
```

### ClassificationResult
```python
class ClassificationResult(BaseModel):
    is_important: bool            # Gate for researcher agent
    confidence: float             # 0.0 to 1.0
    category: ImportanceCategory  # Topic category enum
    reasoning: str                # Explanation

    @classmethod
    def analyze(cls, category, confidence=0.9, reasoning="")
    @classmethod
    def skip(cls, category=OTHER, reasoning="...")
```

### ResearchReport
```python
class ResearchReport(BaseModel):
    summary: str                  # Detailed analysis (~600-1000 words)
    thought: str                  # Source analysis notes
    key_points: list[str]         # 3-5 bullet points
    related_topics: list[str]     # Follow-up topics

    @classmethod
    def empty(cls)                # For error cases
```

## Database Schema

```sql
-- Main stories table
CREATE TABLE stories (
    hash TEXT PRIMARY KEY,         -- 16-char dedup hash
    title TEXT NOT NULL,           -- Story headline
    pub_date INTEGER NOT NULL,     -- Unix timestamp
    processed_at INTEGER NOT NULL, -- When we processed it
    is_important INTEGER DEFAULT 0,-- 0=skipped, 1=analyzed
    summary TEXT,                  -- Analysis summary
    thought TEXT,                  -- Analysis notes
    source_url TEXT                -- RSS feed URL
);

-- Indexes
CREATE INDEX idx_processed ON stories(processed_at);
CREATE INDEX idx_important ON stories(is_important);
```

## Extending the Pipeline

### Adding a New Tool

1. Create `tools/new_tool.py`:
```python
async def my_tool(query: str) -> MyResult:
    """Tool description for the agent."""
    # Implementation
    return MyResult(...)
```

2. Register in `agents/researcher.py`:
```python
@agent.tool
async def my_tool(ctx: RunContext[ResearchContext], query: str) -> str:
    """Docstring shown to model - explain when to use this tool."""
    result = await my_tool_impl(query)
    return result.summary  # Return string for model
```

### Adding a New RSS Feed

Edit `config.py`:
```python
DEFAULT_RSS_URLS = [
    # ... existing feeds ...
    "https://example.com/feed.xml",  # Your new feed
]
```

### Implementing Web Search

Edit `tools/search.py` - see the integration guide in the module docstring.

## Error Handling

- **Classifier failures**: Default to `is_important=True` (fail-safe)
- **Researcher failures**: Return empty report, log error
- **Feed failures**: Skip feed, continue with others
- **Database errors**: Log and raise
- **Notification errors**: Log and continue

## Logging

### Overview

The pipeline uses Python's standard `logging` module with enhanced features:
- **Dual output**: Console (configurable level) + rotating file (always DEBUG)
- **Structured logging**: Optional JSON format for log aggregation
- **Context propagation**: Run ID automatically included in all log messages
- **Trace correlation**: Integration with Logfire distributed tracing

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Console verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `LOG_FORMAT` | `text` | Output format: `text` (human-readable) or `json` (structured) |
| `LOG_BACKUP_COUNT` | `30` | Number of rotated log files to keep |
| `LOG_MAX_BYTES` | `0` | Max file size in bytes (0 = daily time-based rotation) |
| `LOG_DIR` | `log` | Directory for log files |

### Log Format

**Text format** (default):
```
14:23:45 [INFO] [a1b2c3d4] pipeline: Pipeline started | feeds=17
```

**JSON format** (`LOG_FORMAT=json`):
```json
{"timestamp": "2024-01-15T14:23:45Z", "level": "INFO", "logger": "pipeline", "message": "Pipeline started | feeds=17", "run_id": "a1b2c3d4"}
```

### Log Levels

| Level | Used For |
|-------|----------|
| DEBUG | Detailed operations: classifications, fetches, database queries |
| INFO | Pipeline lifecycle: start, completion, important stories |
| WARNING | Recoverable issues: feed timeouts, webhook failures, SSL retries |
| ERROR | Failures with stack traces: API errors, analysis failures |

### Key Log Messages

```bash
# Pipeline lifecycle
"Pipeline started | feeds=17"
"Fetch complete | total=42 new=5"
"Classification complete | important=3 skip=2"
"Story analyzed | hash=abc123 title=..."
"Pipeline done | duration=12.5s important=3 notified=3 errors=0"

# Errors (with stack traces)
"Classification failed for '...': API error"
"Analysis failed for '...': timeout"
"Webhook failed | status=500 title=..."
```

### Run ID Context

Each pipeline run generates a unique 8-character run ID that's automatically included in all log messages. This enables filtering logs for a specific run:

```bash
# Filter logs by run ID
grep "a1b2c3d4" log/photon.log
```

### Graceful Degradation

If the log directory is not writable, the system falls back to console-only logging with a warning message. This ensures the pipeline continues running even with filesystem issues.

## Development

```bash
# Syntax check
python -m py_compile main.py pipeline.py

# Import test
python -c "from pipeline import Pipeline; print('OK')"

# Run with debug logging
python main.py run -v

# Check database
sqlite3 news.db "SELECT COUNT(*) FROM stories"
```

## Common Tasks

### Check Pipeline Status
```bash
python main.py status
```

### View Recent Important Stories
```bash
python main.py recent --hours 24
```

### Force Re-analyze a Story
```bash
python main.py analyze --title "Story Title" --force
```

### Clear Old Records
Records are automatically pruned based on `PRUNE_AFTER_DAYS` (default: 30).

## Git Conventions

- Do not add Claude as a co-author in commit messages
