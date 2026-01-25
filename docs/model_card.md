# Model Card

## Overview
This system is a news analysis pipeline with three ML components:
- Classifier: fast importance triage using an LLM (local MLX by default).
- Retriever: hybrid RAG (BM25 + vector search with RRF).
- Researcher: Gemini with built-in web search grounding.

## Intended Use
Monitor tech and AI RSS feeds, flag important items, and produce structured
analysis reports for internal research and briefing workflows.

## Data Sources
- RSS feeds listed in `config.py`
- Article content fetched via `tools/fetch.py`
- Historical context stored in `news.db` (summaries + embeddings)
- Optional Gemini web grounding during analysis

## Evaluation
See `eval/README.md` for classification and retrieval evaluation harnesses.
Store labeled datasets under `data/labels/` using `data/label_schema.json`.

### Current Snapshot (Seed Labels, 2026-01-24)
- **Classifier** (`data/labels/seed.jsonl`, n=160): accuracy 0.7438, precision 0.7583, recall 0.8835, F1 0.8161, category accuracy 0.50.
- **Retrieval** (`eval/queries.jsonl`, self-retrieval baseline): MRR 1.00, Recall@5 1.00.
- **Note**: Seed labels include heuristic annotations; treat these as smoke-test metrics, not production-quality evaluation.

### Gold Snapshot (Manual Labels, 2026-01-24)
- **Classifier** (`data/labels/gold.jsonl`, n=20): accuracy 0.80, precision 0.7333, recall 1.00, F1 0.8462, category accuracy 0.35.
- **Note**: Gold set is small; use it for sanity checks until more manual labels are added.

## Limitations
- Classification relies on headline/summary context and may miss nuance.
- Web grounding depends on external availability and may return partial context.
- Deduplication is hour-based; republished stories within the same hour may be merged.
- Embeddings are English-first (BGE-small-en-v1.5); non-English retrieval may degrade.

## Known Failure Modes
- False positives when marketing copy resembles product launches.
- False negatives for subtle policy/regulatory updates with vague titles.
- Retrieval gaps if embeddings are missing or if summaries are empty.
- Analysis quality drops when article fetches fail.

## Monitoring and Drift Signals
- Important rate by feed and overall (sudden shifts can indicate drift).
- Retrieval hit rate from `eval/retrieval_eval.py` on a fixed query set.
- Coverage of embeddings for important stories.
- API error rates and latency percentiles from logs.

## Responsible Use
Outputs are for research assistance and require human review before
external publication or decision-making.
