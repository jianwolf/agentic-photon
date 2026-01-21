"""Database operations for the Photon news analysis pipeline."""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from models.story import Story
from models.research import Analysis


class Database:
    """SQLite database for storing processed stories."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS stories (
        hash TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        pub_date INTEGER NOT NULL,
        processed_at INTEGER NOT NULL,
        is_important INTEGER DEFAULT 0,
        summary TEXT,
        thought TEXT,
        source_url TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_processed ON stories(processed_at);
    CREATE INDEX IF NOT EXISTS idx_important ON stories(is_important);
    """

    def __init__(self, path: Path | str):
        """Initialize database connection."""
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Run any necessary migrations."""
        cursor = self.conn.execute("PRAGMA table_info(stories)")
        columns = {row["name"] for row in cursor.fetchall()}

        # Add missing columns from schema evolution
        if "source_url" not in columns:
            self.conn.execute("ALTER TABLE stories ADD COLUMN source_url TEXT")
            self.conn.commit()

    def seen_hashes(self, hashes: set[str]) -> set[str]:
        """Check which hashes already exist in the database.

        Args:
            hashes: Set of story hashes to check

        Returns:
            Set of hashes that already exist in the database
        """
        if not hashes:
            return set()

        placeholders = ",".join("?" * len(hashes))
        query = f"SELECT hash FROM stories WHERE hash IN ({placeholders})"
        cursor = self.conn.execute(query, list(hashes))
        return {row["hash"] for row in cursor.fetchall()}

    def save(
        self,
        story: Story,
        analysis: Analysis,
        commit: bool = True
    ) -> None:
        """Save a story and its analysis to the database.

        Args:
            story: The story to save
            analysis: Analysis results for the story
            commit: Whether to commit immediately (False for batch operations)
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO stories
            (hash, title, pub_date, processed_at, is_important, summary, thought, source_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                story.hash,
                story.title,
                int(story.pub_date.timestamp()),
                int(time.time()),
                int(analysis.is_important),
                analysis.summary,
                analysis.thought,
                story.source_url,
            ),
        )
        if commit:
            self.conn.commit()

    def commit(self) -> None:
        """Commit pending changes."""
        self.conn.commit()

    def prune(self, days: int) -> int:
        """Delete records older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of records deleted
        """
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
        cursor = self.conn.execute(
            "DELETE FROM stories WHERE processed_at < ?",
            (cutoff,)
        )
        self.conn.commit()
        return cursor.rowcount

    def recent(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get important stories from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of story records as dictionaries
        """
        cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
        cursor = self.conn.execute(
            """
            SELECT hash, title, pub_date, processed_at, is_important,
                   summary, thought, source_url
            FROM stories
            WHERE is_important = 1 AND processed_at >= ?
            ORDER BY processed_at DESC
            """,
            (cutoff,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def stats(self) -> dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with total and important story counts
        """
        cursor = self.conn.execute(
            "SELECT COUNT(*) as total, SUM(is_important) as important FROM stories"
        )
        row = cursor.fetchone()
        return {
            "total": row["total"] or 0,
            "important": row["important"] or 0,
        }

    def get_story(self, hash: str) -> dict[str, Any] | None:
        """Get a story by its hash.

        Args:
            hash: Story hash

        Returns:
            Story record as dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM stories WHERE hash = ?",
            (hash,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> "Database":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
