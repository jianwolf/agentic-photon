"""Database operations for the Photon news analysis pipeline.

This module provides SQLite-based storage for processed stories and their
analysis results. It handles deduplication, pruning, and statistics.

Database Schema:
    stories table:
        - hash (TEXT, PK): 16-char dedup hash from Story.hash
        - title (TEXT): Story title
        - pub_date (INTEGER): Publication timestamp (Unix epoch)
        - processed_at (INTEGER): Processing timestamp (Unix epoch)
        - is_important (INTEGER): 0/1 flag for importance
        - summary (TEXT): Analysis summary (if important)
        - thought (TEXT): Analysis notes (if important)
        - source_url (TEXT): RSS feed URL

    Indexes:
        - idx_processed: For time-based queries and pruning
        - idx_important: For filtering important stories

Features:
    - WAL mode for concurrent read/write access
    - Automatic schema migration for new columns
    - Batch operations with deferred commits
    - Context manager support for auto-cleanup
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from models.story import Story
from models.research import Analysis

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for storing processed stories.

    Provides storage and retrieval for the news analysis pipeline.
    Stories are deduplicated by their hash and can be queried by
    recency and importance.

    Example:
        >>> with Database("news.db") as db:
        ...     db.save(story, analysis)
        ...     recent = db.recent(hours=24)
    """

    # SQL schema for the stories table and indexes
    SCHEMA = """
    -- Main stories table: one row per processed story
    CREATE TABLE IF NOT EXISTS stories (
        hash TEXT PRIMARY KEY,           -- 16-char dedup hash
        title TEXT NOT NULL,             -- Story headline
        pub_date INTEGER NOT NULL,       -- Publication time (Unix epoch)
        processed_at INTEGER NOT NULL,   -- When we processed it (Unix epoch)
        is_important INTEGER DEFAULT 0,  -- 0=skipped, 1=analyzed
        summary TEXT,                    -- Analysis summary (NULL if not important)
        thought TEXT,                    -- Analysis notes (NULL if not important)
        source_url TEXT                  -- RSS feed URL
    );

    -- Index for time-based queries (recent stories, pruning)
    CREATE INDEX IF NOT EXISTS idx_processed ON stories(processed_at);

    -- Index for filtering by importance
    CREATE INDEX IF NOT EXISTS idx_important ON stories(is_important);
    """

    def __init__(self, path: Path | str):
        """Initialize database connection.

        Creates the database file if it doesn't exist and sets up
        the schema. Uses WAL mode for better concurrent access.

        Args:
            path: Path to SQLite database file
        """
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row  # Enable dict-like row access

        # WAL mode allows concurrent readers during writes
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        logger.debug("Database initialized | path=%s", self.path)

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist.

        Also runs any pending migrations for schema evolution.
        """
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Run schema migrations for backwards compatibility.

        Adds columns that were added in later versions without
        requiring a database rebuild.
        """
        cursor = self.conn.execute("PRAGMA table_info(stories)")
        columns = {row["name"] for row in cursor.fetchall()}

        # Migration: source_url column added after initial release
        if "source_url" not in columns:
            self.conn.execute("ALTER TABLE stories ADD COLUMN source_url TEXT")
            self.conn.commit()
            logger.info("Database migrated | added column=source_url")

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
        logger.debug("Story saved | hash=%s important=%s", story.hash, analysis.is_important)

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
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("Database pruned | deleted=%d days=%d", deleted, days)
        return deleted

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
