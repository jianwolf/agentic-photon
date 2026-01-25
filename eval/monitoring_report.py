#!/usr/bin/env python3
"""Generate a lightweight monitoring snapshot from the SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _fetch_counts(conn: sqlite3.Connection, start_ts: int, end_ts: int | None = None) -> dict[str, int]:
    if end_ts is None:
        cursor = conn.execute(
            """
            SELECT COUNT(*) as total, SUM(is_important) as important
            FROM stories
            WHERE processed_at >= ?
            """,
            (start_ts,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT COUNT(*) as total, SUM(is_important) as important
            FROM stories
            WHERE processed_at >= ? AND processed_at < ?
            """,
            (start_ts, end_ts),
        )
    row = cursor.fetchone()
    return {
        "total": row[0] or 0,
        "important": row[1] or 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate monitoring snapshot.")
    parser.add_argument("--db-path", default="news.db", help="SQLite database path.")
    parser.add_argument("--days", type=int, default=7, help="Lookback window.")
    parser.add_argument("--top-sources", type=int, default=5, help="Top sources to show.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    now = datetime.now(timezone.utc)
    current_start = int((now - timedelta(days=args.days)).timestamp())
    previous_start = int((now - timedelta(days=2 * args.days)).timestamp())
    previous_end = current_start

    with sqlite3.connect(str(db_path)) as conn:
        current = _fetch_counts(conn, current_start)
        previous = _fetch_counts(conn, previous_start, previous_end)

        cursor = conn.execute(
            """
            SELECT source_url,
                   COUNT(*) as total,
                   SUM(is_important) as important
            FROM stories
            WHERE processed_at >= ?
            GROUP BY source_url
            ORDER BY total DESC
            LIMIT ?
            """,
            (current_start, args.top_sources),
        )
        top_sources = [
            {
                "source_url": row[0],
                "total": row[1],
                "important": row[2] or 0,
            }
            for row in cursor.fetchall()
        ]

        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM story_embeddings e
            JOIN stories s ON e.hash = s.hash
            WHERE s.is_important = 1 AND s.processed_at >= ?
            """,
            (current_start,),
        )
        embedded_count = cursor.fetchone()[0] or 0

    current_rate = (current["important"] / current["total"]) if current["total"] else 0.0
    previous_rate = (previous["important"] / previous["total"]) if previous["total"] else 0.0

    report = {
        "window_days": args.days,
        "current": {
            "total": current["total"],
            "important": current["important"],
            "important_rate": round(current_rate, 4),
            "embedded_important": embedded_count,
        },
        "previous": {
            "total": previous["total"],
            "important": previous["important"],
            "important_rate": round(previous_rate, 4),
        },
        "top_sources": top_sources,
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
