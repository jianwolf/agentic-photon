# Data and Labeling

This folder contains scripts and schema for building labeled datasets for
classification and retrieval evaluation.

## Label Pool Collection

Collect a fresh labeling pool from the configured RSS feeds:

```bash
python data/collect_label_pool.py --max-age-hours 168 --max-items 200 --out data/labels/pool.jsonl
```

The output is JSONL that follows `data/label_schema.json`. Set `label` to
`true` or `false` and optionally fill `category` and `notes`.

## Label Schema

`data/label_schema.json` defines required and optional fields:

- `id`: stable identifier (typically `Story.hash`)
- `title`, `description`, `source_url`, `article_url`, `pub_date`
- `label`: boolean (or `null` for unlabeled pools)
- `category`: optional string (use `ImportanceCategory` values)
- `notes`: optional short rationale

## Seed Labels (Quickstart)

`data/labels/seed.jsonl` is a starter set that mixes a small number of
manual labels with heuristic labels for quick iteration. Treat it as a
smoke-test dataset and replace it with human-reviewed labels for
production evaluation.

To extract a manual-only gold subset:

```bash
python data/extract_gold_labels.py --input data/labels/seed.jsonl --out data/labels/gold.jsonl
```

## Deterministic Splits

Create stable train/dev/test splits:

```bash
python data/split_labels.py --input data/labels/pool.jsonl --out-dir data/labels/splits
```

## Retrieval Eval Set

Create a self-retrieval query set from your local database:

```bash
python data/build_retrieval_queries.py --db-path news.db --out eval/queries_title.jsonl --query-mode title
python data/build_retrieval_queries.py --db-path news.db --out eval/queries_title_summary.jsonl --query-mode title_summary
```

This creates query records where each query is a story title and the relevant
set contains the story itself (a sanity-check baseline).
