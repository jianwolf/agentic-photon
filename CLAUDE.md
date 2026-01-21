# Agentic Photon - Developer Guide

## Overview

Agentic Photon is a PydanticAI-powered news analysis pipeline that:
1. Fetches RSS feeds from curated tech/AI sources
2. Classifies stories by importance using a fast classifier agent
3. Performs deep analysis on important stories using a researcher agent
4. Generates markdown reports and sends notifications

## Architecture

```
RSS Feeds → Async Fetch → Dedup → Classifier → [Important?]
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                            Researcher Agent                   Save Empty
                                    │
                        ┌───────────┼───────────┐
                        ↓           ↓           ↓
                    Search      Fetch URL   Query DB
                        │           │           │
                        └───────────┼───────────┘
                                    ↓
                            Research Report → Notifications
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `pipeline.py` | Main orchestration |
| `config.py` | Environment-based configuration |
| `database.py` | SQLite operations |
| `feeds.py` | Async RSS fetching |
| `notifications.py` | Reports and alerts |

### Agents

| File | Purpose |
|------|---------|
| `agents/classifier.py` | `ClassifierAgent` - fast importance classification |
| `agents/researcher.py` | `ResearcherAgent` - deep analysis with tools |

### Models

| File | Purpose |
|------|---------|
| `models/story.py` | `Story` - RSS feed item |
| `models/classification.py` | `ClassificationResult`, `ImportanceCategory` |
| `models/research.py` | `ResearchReport`, `Analysis` |

### Tools

| File | Purpose |
|------|---------|
| `tools/search.py` | Web search (placeholder) |
| `tools/fetch.py` | Article content extraction |
| `tools/database.py` | Story history queries |

## Pipeline Flow

1. **Fetch**: `fetch_all_feeds()` concurrently fetches all RSS feeds
2. **Dedup**: `db.seen_hashes()` filters already-processed stories
3. **Classify**: `ClassifierAgent.classify_batch()` determines importance
4. **Analyze**: `ResearcherAgent.analyze_batch()` processes important stories
5. **Save**: `db.save()` stores all results
6. **Notify**: `notify_batch()` sends reports/webhooks

## Configuration

All configuration via environment variables:

```bash
# Required
GEMINI_API_KEY=...

# Models
CLASSIFIER_MODEL=google-gla:gemini-2.0-flash
RESEARCHER_MODEL=google-gla:gemini-2.0-flash

# Pipeline
LANGUAGE=zh          # zh or en
MAX_AGE_HOURS=720    # 30 days
POLL_INTERVAL_SECONDS=300

# Output
DB_PATH=news.db
REPORTS_DIR=reports
LOG_DIR=log

# Optional notifications
NOTIFICATION_WEBHOOK_URL=
ALERTS_FILE=

# Optional features
ENABLE_MEMORY=false   # ChromaDB
ENABLE_LOGFIRE=false  # Tracing
```

## CLI Usage

```bash
# Run pipeline once
python main.py run

# Run with English output
python main.py run --lang en

# Continuous mode
python main.py run -c

# Custom poll interval
python main.py run -c --interval 600

# Check configuration
python main.py status

# View recent stories
python main.py recent --hours 48

# Analyze specific story
python main.py analyze --title "Story Title" --force
```

## Data Models

### Story
```python
class Story(BaseModel):
    title: str
    description: str
    pub_date: datetime
    source_url: str

    @property
    def hash(self) -> str: ...      # 16-char dedup hash
    @property
    def publishers(self) -> list[str]: ...  # Extracted from HTML
```

### ClassificationResult
```python
class ClassificationResult(BaseModel):
    is_important: bool
    confidence: float  # 0-1
    category: ImportanceCategory
    reasoning: str
```

### ResearchReport
```python
class ResearchReport(BaseModel):
    summary: str
    thought: str
    key_points: list[str]
    related_topics: list[str]
```

## Database Schema

```sql
CREATE TABLE stories (
    hash TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    pub_date INTEGER NOT NULL,
    processed_at INTEGER NOT NULL,
    is_important INTEGER DEFAULT 0,
    summary TEXT,
    thought TEXT,
    source_url TEXT
);
```

## Extending

### Adding a New Tool

1. Create `tools/new_tool.py`:
```python
async def my_tool(param: str) -> str:
    """Tool description for the agent."""
    # Implementation
    return result
```

2. Register in `agents/researcher.py`:
```python
@agent.tool
async def my_tool(ctx: RunContext[ResearchContext], param: str) -> str:
    return await my_tool_impl(param)
```

### Adding a New Feed

Edit `config.py`:
```python
DEFAULT_RSS_URLS = [
    # ... existing feeds ...
    "https://example.com/feed.xml",
]
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
