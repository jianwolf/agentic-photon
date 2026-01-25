# Ablation Plan and Results

This document tracks retrieval and prompt ablations. Keep runs reproducible by
recording the dataset version, model IDs, and commit hash.

## Retrieval Ablations

| Variant | Change | Metric | Result |
| --- | --- | --- | --- |
| BM25 only | Disable vector search | MRR, Recall@5 | MRR 0.8889, Recall@5 0.8889 (27 queries) |
| Vector only | Disable BM25 | MRR, Recall@5 | MRR 1.00, Recall@5 1.00 (27 queries) |
| Hybrid (RRF) | BM25 + Vector | MRR, Recall@5 | MRR 1.00, Recall@5 1.00 (self-retrieval baseline, 27 queries) |
| Query truncation | Title only vs title+summary | MRR, Recall@5 | Title-only: MRR 1.00, Recall@5 1.00 (27 queries); Title+summary: MRR 1.00, Recall@5 1.00 (27 queries) |

## Classifier Prompt Ablations

| Variant | Change | Metric | Result |
| --- | --- | --- | --- |
| Baseline | Current prompt | F1, precision/recall | F1 0.8161, P 0.7583, R 0.8835 (seed labels, n=160) |
| No CoT | Remove step-by-step section | F1, precision/recall | TBD |
| Higher temp | Temperature 0.3 | F1, variance | TBD |

## Baseline Details (2026-01-24)

- **Classifier**: `google-gla:gemini-3-flash-preview` on `data/labels/seed.jsonl` (manual + heuristic labels).
- **Retrieval**: `eval/queries.jsonl` self-retrieval set built from `news.db`.
- **Note**: Self-retrieval is a sanity-check baseline; it will overestimate MRR/Recall.

## How to Run

1. Prepare labeled data using `data/README.md`.
2. Run `eval/classification_eval.py` for classifier ablations.
3. Run `eval/retrieval_eval.py` for retrieval ablations.
4. Record results here with dataset and model metadata.
