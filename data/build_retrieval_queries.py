#!/usr/bin/env python3
"""Build a simple retrieval query set from the local database."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create retrieval queries from news.db.")
    parser.add_argument("--db-path", default="news.db", help="SQLite database path.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window.")
    parser.add_argument("--limit", type=int, default=50, help="Max queries to sample.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--out", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--query-mode",
        choices=["title", "title_summary"],
        default="title",
        help="Query construction mode.",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=300,
        help="Max summary chars to append when using title_summary.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    cutoff = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp())
    rows: list[tuple[str, str, str]] = []

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute(
            """
            SELECT hash, title, summary
            FROM stories
            WHERE is_important = 1 AND processed_at >= ?
            """,
            (cutoff,),
        )
        rows = [(row[0], row[1], row[2] or "") for row in cursor.fetchall()]

    if not rows:
        raise SystemExit("No important stories found for the requested window.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for story_hash, title, summary in rows:
            query = title
            if args.query_mode == "title_summary" and summary:
                snippet = " ".join(summary.split())
                if args.summary_chars > 0:
                    snippet = snippet[: args.summary_chars]
                query = f"{title} {snippet}"

            record = {
                "id": story_hash,
                "query": query,
                "relevant_hashes": [story_hash],
                "notes": f"Self-retrieval baseline ({args.query_mode}).",
                "query_mode": args.query_mode,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} queries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
