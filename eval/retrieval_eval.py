#!/usr/bin/env python3
"""Evaluate hybrid retrieval using labeled query sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from database import Database
from embeddings import encode_text


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _reciprocal_rank(results: list[str], relevant: set[str]) -> float:
    for rank, item_id in enumerate(results, 1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def _recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in results[:k]) else 0.0


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval.")
    parser.add_argument("--db-path", default="news.db", help="SQLite database path.")
    parser.add_argument("--queries", required=True, help="Queries JSONL file.")
    parser.add_argument("--limit", type=int, default=10, help="Top N results to retrieve.")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window for retrieval.")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "bm25", "vector"],
        default="hybrid",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument("--out-details", help="Write per-query metrics JSONL.")
    args = parser.parse_args()

    queries = _load_jsonl(Path(args.queries))
    if not queries:
        raise SystemExit("No queries found.")

    db = Database(args.db_path)
    details: list[dict] = []
    total_rr = 0.0
    total_recall = 0.0
    evaluated = 0

    try:
        for record in queries:
            query = record.get("query")
            relevant = record.get("relevant_hashes") or record.get("relevant_ids") or []
            if not query or not relevant:
                continue

            relevant_set = {str(item) for item in relevant}
            if args.mode == "bm25":
                bm25_results = db._bm25_search(query, limit=args.limit, days=args.days)
                result_ids = [hash_id for hash_id, _ in bm25_results]
            elif args.mode == "vector":
                query_embedding = encode_text(query)
                vector_results = db._vector_search(
                    query_embedding=query_embedding,
                    limit=args.limit,
                    days=args.days,
                )
                result_ids = [hash_id for hash_id, _ in vector_results]
            else:
                query_embedding = encode_text(query)
                results = db.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    limit=args.limit,
                    days=args.days,
                )
                result_ids = [story.hash for story in results.stories]

            rr = _reciprocal_rank(result_ids, relevant_set)
            recall = _recall_at_k(result_ids, relevant_set, args.k)

            details.append(
                {
                    "id": record.get("id"),
                    "query": query,
                    "reciprocal_rank": rr,
                    "recall_at_k": recall,
                    "results": result_ids,
                    "relevant": list(relevant_set),
                    "mode": args.mode,
                }
            )

            total_rr += rr
            total_recall += recall
            evaluated += 1
    finally:
        db.close()

    if evaluated == 0:
        raise SystemExit("No queries evaluated. Check your query file.")

    metrics = {
        "queries": evaluated,
        "mrr": round(total_rr / evaluated, 4),
        "recall_at_k": round(total_recall / evaluated, 4),
        "k": args.k,
        "limit": args.limit,
        "days": args.days,
        "mode": args.mode,
    }

    print(json.dumps(metrics, indent=2))

    if args.out_details:
        _write_jsonl(Path(args.out_details), details)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
