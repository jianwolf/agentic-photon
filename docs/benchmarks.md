# Benchmarks

Record latency and throughput measurements here. Use `eval/benchmark.py` and
note hardware details to keep results comparable.

## Embeddings

| Date | Hardware | Model | Items | Avg ms/item |
| --- | --- | --- | --- | --- |
| 2026-01-24 | Darwin arm64 (local) | BAAI/bge-small-en-v1.5 | 20 | 299.6 |

## Retrieval (Hybrid RAG)

| Date | Hardware | DB Size | Avg ms/query | Limit |
| --- | --- | --- | --- | --- |
| 2026-01-24 | Darwin arm64 (local) | 92 stories | 44.4 | 5 |

## Classification

| Date | Hardware | Model | Avg ms/story | Notes |
| --- | --- | --- | --- | --- |
| 2026-01-24 | Darwin arm64 (local) | google-gla:gemini-3-flash-preview | 3043.9 | n=50, concurrency=4 |
| 2026-01-24 | Darwin arm64 (local) | mlx-community/Ministral-3-3B-Instruct-2512 | 5576.4 | n=30, concurrency=2 |
