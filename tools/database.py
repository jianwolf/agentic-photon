"""Database tool for querying story history."""

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StoryHistory:
    """Results from querying related stories."""

    topic: str
    stories: list[dict] = field(default_factory=list)
    days: int = 7

    @property
    def summary(self) -> str:
        """Format as readable summary."""
        if not self.stories:
            return f"No related stories found for: {self.topic}"

        lines = [f"Related stories for '{self.topic}' (last {self.days} days):\n"]
        for s in self.stories[:5]:
            lines.append(f"- {s['title']}")
            lines.append(f"  Date: {s['pub_date']}")
            if s.get("summary"):
                preview = s["summary"][:200] + "..." if len(s["summary"]) > 200 else s["summary"]
                lines.append(f"  {preview}")
            lines.append("")
        return "\n".join(lines)


async def query_related_stories(
    topic: str,
    db_path: Path | str,
    days: int = 7,
    limit: int = 10,
) -> StoryHistory:
    """Query database for stories related to a topic.

    Uses simple keyword matching on title and summary.
    For production, consider FTS5 or vector similarity.

    Args:
        topic: Keywords to search
        db_path: SQLite database path
        days: Days to look back
        limit: Max results

    Returns:
        StoryHistory with matching stories
    """
    logger.debug(f"Querying related stories: {topic}")

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_ts = int(cutoff.timestamp())

        # Build keyword search
        keywords = topic.lower().split()
        if keywords:
            conditions = " OR ".join(
                "(LOWER(title) LIKE ? OR LOWER(summary) LIKE ?)"
                for _ in keywords
            )
            params = [cutoff_ts]
            for kw in keywords:
                params.extend([f"%{kw}%", f"%{kw}%"])
            params.append(limit)

            query = f"""
                SELECT hash, title, pub_date, summary, source_url
                FROM stories
                WHERE is_important = 1
                  AND processed_at >= ?
                  AND ({conditions})
                ORDER BY processed_at DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT hash, title, pub_date, summary, source_url
                FROM stories
                WHERE is_important = 1 AND processed_at >= ?
                ORDER BY processed_at DESC
                LIMIT ?
            """
            params = [cutoff_ts, limit]

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        stories = [
            {
                "hash": row["hash"],
                "title": row["title"],
                "pub_date": datetime.fromtimestamp(row["pub_date"], timezone.utc).strftime("%Y-%m-%d"),
                "summary": row["summary"] or "",
                "source_url": row["source_url"] or "",
            }
            for row in rows
        ]

        return StoryHistory(topic=topic, stories=stories, days=days)

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return StoryHistory(topic=topic, stories=[], days=days)
