# Repository Guidelines

## Project Structure & Module Organization
Core orchestration lives in `main.py` (CLI) and `pipeline.py` (run loop). Configuration is in `config.py`, persistence and retrieval in `database.py` and `embeddings.py`, and I/O utilities in `feeds.py`, `notifications.py`, and `mlx_server.py`. Agent logic is under `agents/` (classifier and researcher), Pydantic models are in `models/`, and supporting helpers are in `tools/`. Observability integrations live in `observability/`. Generated artifacts include `reports/`, `log/`, and the local SQLite `news.db`; avoid committing these.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Run once: `python main.py run`
- Continuous mode: `python main.py run -c --interval 600`
- Status/debug: `python main.py status`, `python main.py run -v`
- Dev checks: `python -m py_compile main.py pipeline.py`, `python -c "from pipeline import Pipeline; print('OK')"`

## Coding Style & Naming Conventions
Use Python 3 style with 4-space indentation, docstrings for public functions, and type hints where practical. Follow existing naming: `snake_case` for functions/variables, `PascalCase` for classes, and `ALL_CAPS` for constants. There is no enforced formatter; keep changes aligned with current layout and logging patterns in `observability/logging.py`.

## Testing Guidelines
Testing uses `pytest` and `pytest-asyncio` (no coverage threshold is configured). There is currently no `tests/` directory; when adding tests, create `tests/test_*.py` and run `pytest -q`. Prefer unit tests for classification, database queries, and pipeline stages affected by a change.

## Commit & Pull Request Guidelines
Commit messages are short, imperative, and action-focused (examples: "Fix logging issues", "Improve classifier consistency"). Do not add Claude as a co-author. PRs should include a concise summary, testing notes, and any config or environment variable changes (for example, `GEMINI_API_KEY`). If you change architecture or major behavior, update `README.md` to explain trade-offs and rationale.
