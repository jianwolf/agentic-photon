#!/usr/bin/env python3
"""Lightweight benchmarks for embeddings and retrieval latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from database import Database
from embeddings import encode_text, encode_texts


def _load_texts(labels_path: Path, max_items: int) -> list[str]:
    texts: list[str] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            title = record.get("title") or ""
            description = record.get("description") or ""
            text = f"{title} {description}".strip()
            if text:
                texts.append(text)
            if max_items and len(texts) >= max_items:
                break
    return texts


def _synthetic_texts(count: int) -> list[str]:
    return [f"Synthetic text sample {i} for embedding benchmarks." for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run embedding and retrieval benchmarks.")
    parser.add_argument("--labels", help="Optional JSONL file for real text samples.")
    parser.add_argument("--num-texts", type=int, default=200, help="Number of texts.")
    parser.add_argument("--db-path", default="news.db", help="SQLite database path.")
    parser.add_argument("--rag-queries", type=int, default=50, help="Number of retrieval queries.")
    parser.add_argument("--limit", type=int, default=5, help="Hybrid search result limit.")
    parser.add_argument("--out", help="Write metrics JSON.")
    args = parser.parse_args()

    if args.labels:
        texts = _load_texts(Path(args.labels), args.num_texts)
    else:
        texts = _synthetic_texts(args.num_texts)

    if not texts:
        raise SystemExit("No texts available for benchmarking.")

    # Embedding benchmark
    start = time.perf_counter()
    _ = encode_texts(texts)
    embed_duration = time.perf_counter() - start
    embed_avg_ms = (embed_duration / len(texts)) * 1000

    metrics = {
        "embedding": {
            "items": len(texts),
            "total_s": round(embed_duration, 4),
            "avg_ms": round(embed_avg_ms, 4),
        }
    }

    # Retrieval benchmark (optional)
    db_path = Path(args.db_path)
    if db_path.exists():
        db = Database(db_path)
        try:
            query_texts = texts[: args.rag_queries]
            start = time.perf_counter()
            for query in query_texts:
                query_embedding = encode_text(query)
                db.hybrid_search(query=query, query_embedding=query_embedding, limit=args.limit, days=30)
            rag_duration = time.perf_counter() - start
            rag_avg_ms = (rag_duration / len(query_texts)) * 1000
            metrics["retrieval"] = {
                "queries": len(query_texts),
                "total_s": round(rag_duration, 4),
                "avg_ms": round(rag_avg_ms, 4),
                "limit": args.limit,
            }
        finally:
            db.close()
    else:
        metrics["retrieval"] = {"skipped": True, "reason": f"Database not found: {db_path}"}

    output = json.dumps(metrics, indent=2)
    print(output)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
