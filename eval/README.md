# Evaluation

This directory contains evaluation and benchmarking utilities for the
classification and retrieval components.

## Classification Evaluation

Run the classifier against labeled JSONL and compute metrics:

```bash
python eval/classification_eval.py --labels data/labels/splits/dev.jsonl \
  --out-predictions eval/preds_dev.jsonl
```

To use a local MLX model:

```bash
python eval/classification_eval.py --labels data/labels/splits/dev.jsonl \
  --mlx-model mlx-community/Ministral-3-3B-Instruct-2512 --mlx-port 8080
```

## Retrieval Evaluation

Build a query set from your database, then evaluate hybrid search:

```bash
python data/build_retrieval_queries.py --db-path news.db --out eval/queries.jsonl
python eval/retrieval_eval.py --db-path news.db --queries eval/queries.jsonl --limit 10 --k 5

# Mode-specific ablations
python eval/retrieval_eval.py --db-path news.db --queries eval/queries.jsonl --mode bm25
python eval/retrieval_eval.py --db-path news.db --queries eval/queries.jsonl --mode vector
```

## Benchmarks

```bash
python eval/benchmark.py --labels data/labels/sample.jsonl --db-path news.db --out eval/benchmarks.json
```

Classifier latency:

```bash
python eval/classifier_benchmark.py --labels data/labels/seed.jsonl --max-items 50 --out eval/classifier_benchmark.json
```

## Monitoring Snapshot

```bash
python eval/monitoring_report.py --db-path news.db --days 7
```
