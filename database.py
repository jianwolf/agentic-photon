"""Database operations for the Photon news analysis pipeline.

This module provides SQLite-based storage for processed stories and their
analysis results. It handles deduplication, pruning, hybrid RAG search,
and statistics.

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

    stories_fts (FTS5 virtual table):
        - BM25 full-text search on title and summary
        - Indexes ALL stories for keyword matching

    story_embeddings table:
        - hash (TEXT, PK): Story hash (foreign key to stories)
        - embedding (BLOB): 384-dim float32 vector
        - Only stores embeddings for IMPORTANT stories

Features:
    - WAL mode for concurrent read/write access
    - Automatic schema migration for new columns
    - Hybrid RAG: BM25 (FTS5) + Vector search with RRF fusion
    - Batch operations with deferred commits
    - Context manager support for auto-cleanup
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from models.story import Story
from models.research import Analysis

logger = logging.getLogger(__name__)

# Embedding dimension (must match embeddings.py)
EMBEDDING_DIM = 384


@dataclass
class RelatedStory:
    """A story returned from hybrid search."""
    hash: str
    title: str
    pub_date: str
    summary: str
    score: float
    source: str  # "bm25", "vector", or "both"


@dataclass
class SearchResults:
    """Results from hybrid RAG search."""
    query: str
    stories: list[RelatedStory] = field(default_factory=list)

    def format_context(self, max_stories: int = 5) -> str:
        """Format search results as context string for the researcher.

        Args:
            max_stories: Maximum number of stories to include

        Returns:
            Formatted string with related story context
        """
        if not self.stories:
            return "No related stories found in database."

        lines = [f"Related stories from database (query: '{self.query}'):\n"]
        for i, story in enumerate(self.stories[:max_stories], 1):
            lines.append(f"{i}. {story.title}")
            lines.append(f"   Date: {story.pub_date}")
            if story.summary:
                preview = story.summary[:300] + "..." if len(story.summary) > 300 else story.summary
                lines.append(f"   Summary: {preview}")
            lines.append("")
        return "\n".join(lines)


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Convert numpy embedding to SQLite BLOB."""
    return embedding.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Convert SQLite BLOB to numpy embedding."""
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Assumes vectors are already normalized (which BGE embeddings are).
    """
    return float(np.dot(a, b))


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) for each ranking where item appears.

    Args:
        rankings: List of ranked lists, each containing (id, score) tuples
        k: RRF constant (default 60, standard value)

    Returns:
        Combined ranking as list of (id, rrf_score) tuples, sorted by score desc
    """
    rrf_scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, (item_id, _) in enumerate(ranking, 1):
            if item_id not in rrf_scores:
                rrf_scores[item_id] = 0.0
            rrf_scores[item_id] += 1.0 / (k + rank)

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


class Database:
    """SQLite database for storing processed stories with hybrid RAG support.

    Provides storage and retrieval for the news analysis pipeline.
    Stories are deduplicated by their hash and can be queried by
    recency, importance, and semantic similarity.

    Hybrid Search:
        - BM25 via FTS5: Full-text search on title and summary
        - Vector search: Cosine similarity on embeddings (important stories only)
        - RRF fusion: Combines both rankings for best results

    Example:
        >>> with Database("news.db") as db:
        ...     db.save(story, analysis)
        ...     db.save_embedding(story.hash, embedding)
        ...     results = db.hybrid_search("OpenAI GPT")
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

    -- Compound index for common query pattern: recent important stories
    CREATE INDEX IF NOT EXISTS idx_important_processed ON stories(is_important, processed_at);

    -- Embeddings table for vector search (important stories only)
    CREATE TABLE IF NOT EXISTS story_embeddings (
        hash TEXT PRIMARY KEY,           -- Story hash (FK to stories)
        embedding BLOB NOT NULL          -- 384-dim float32 vector
    );
    """

    # FTS5 schema (separate because it uses different syntax)
    FTS_SCHEMA = """
    CREATE VIRTUAL TABLE IF NOT EXISTS stories_fts USING fts5(
        hash,
        title,
        summary,
        content='stories',
        content_rowid='rowid',
        tokenize='porter unicode61'
    );
    """

    # Triggers to keep FTS5 in sync with stories table
    FTS_TRIGGERS = """
    -- Trigger: Insert into FTS when story is inserted
    CREATE TRIGGER IF NOT EXISTS stories_ai AFTER INSERT ON stories BEGIN
        INSERT INTO stories_fts(rowid, hash, title, summary)
        VALUES (NEW.rowid, NEW.hash, NEW.title, NEW.summary);
    END;

    -- Trigger: Delete from FTS when story is deleted
    CREATE TRIGGER IF NOT EXISTS stories_ad AFTER DELETE ON stories BEGIN
        INSERT INTO stories_fts(stories_fts, rowid, hash, title, summary)
        VALUES ('delete', OLD.rowid, OLD.hash, OLD.title, OLD.summary);
    END;

    -- Trigger: Update FTS when story is updated
    CREATE TRIGGER IF NOT EXISTS stories_au AFTER UPDATE ON stories BEGIN
        INSERT INTO stories_fts(stories_fts, rowid, hash, title, summary)
        VALUES ('delete', OLD.rowid, OLD.hash, OLD.title, OLD.summary);
        INSERT INTO stories_fts(rowid, hash, title, summary)
        VALUES (NEW.rowid, NEW.hash, NEW.title, NEW.summary);
    END;
    """

    path: Path
    conn: sqlite3.Connection

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
        self._init_fts()

    def _init_fts(self) -> None:
        """Initialize FTS5 table and triggers."""
        # Check if FTS table exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='stories_fts'"
        )
        fts_exists = cursor.fetchone() is not None

        if not fts_exists:
            # Create FTS5 table
            self.conn.executescript(self.FTS_SCHEMA)
            self.conn.commit()
            logger.info("Created FTS5 table for full-text search")

            # Populate FTS from existing stories
            self.conn.execute("""
                INSERT INTO stories_fts(rowid, hash, title, summary)
                SELECT rowid, hash, title, summary FROM stories
            """)
            self.conn.commit()
            logger.info("Populated FTS5 index from existing stories")

        # Check if triggers exist
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='stories_ai'"
        )
        triggers_exist = cursor.fetchone() is not None

        if not triggers_exist:
            self.conn.executescript(self.FTS_TRIGGERS)
            self.conn.commit()
            logger.info("Created FTS5 sync triggers")

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

        The FTS5 index is automatically updated via triggers.

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

    def save_embedding(
        self,
        story_hash: str,
        embedding: np.ndarray,
        commit: bool = True
    ) -> None:
        """Save embedding for an important story.

        Args:
            story_hash: Story hash (must exist in stories table)
            embedding: numpy array of shape (384,)
            commit: Whether to commit immediately
        """
        blob = _embedding_to_blob(embedding)
        self.conn.execute(
            "INSERT OR REPLACE INTO story_embeddings (hash, embedding) VALUES (?, ?)",
            (story_hash, blob),
        )
        if commit:
            self.conn.commit()
        logger.debug("Embedding saved | hash=%s dim=%d", story_hash, len(embedding))

    def commit(self) -> None:
        """Commit pending changes."""
        self.conn.commit()

    def prune(self, days: int) -> int:
        """Delete records older than specified days.

        Also removes corresponding embeddings and FTS entries (via triggers).

        Args:
            days: Number of days to keep

        Returns:
            Number of records deleted
        """
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())

        # Get hashes to delete (for embedding cleanup)
        cursor = self.conn.execute(
            "SELECT hash FROM stories WHERE processed_at < ?",
            (cutoff,)
        )
        hashes_to_delete = [row["hash"] for row in cursor.fetchall()]

        # Delete embeddings first
        if hashes_to_delete:
            placeholders = ",".join("?" * len(hashes_to_delete))
            self.conn.execute(
                f"DELETE FROM story_embeddings WHERE hash IN ({placeholders})",
                hashes_to_delete
            )

        # Delete stories (FTS cleanup happens via trigger)
        cursor = self.conn.execute(
            "DELETE FROM stories WHERE processed_at < ?",
            (cutoff,)
        )
        self.conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("Database pruned | deleted=%d days=%d", deleted, days)
        return deleted

    def _bm25_search(
        self,
        query: str,
        limit: int = 20,
        days: int = 30,
    ) -> list[tuple[str, float]]:
        """Search using BM25 via FTS5.

        Args:
            query: Search query
            limit: Maximum results
            days: Days to look back

        Returns:
            List of (hash, bm25_score) tuples, sorted by score desc
        """
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())

        # Escape query for FTS5 syntax:
        # - Quote each term to prevent special char interpretation
        # - Remove characters that break FTS5 even when quoted
        escaped_terms = []
        for term in query.split():
            # Remove problematic chars and quote the term
            clean_term = ''.join(c for c in term if c.isalnum() or c in '-_')
            if clean_term:
                escaped_terms.append(f'"{clean_term}"')

        if not escaped_terms:
            return []

        fts_query = ' '.join(escaped_terms)

        # FTS5 match query with BM25 scoring
        cursor = self.conn.execute(
            """
            SELECT s.hash, bm25(stories_fts) as score
            FROM stories_fts
            JOIN stories s ON stories_fts.hash = s.hash
            WHERE stories_fts MATCH ?
              AND s.processed_at >= ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, cutoff, limit)
        )
        # Note: BM25 returns negative scores (lower is better), so we negate
        return [(row["hash"], -row["score"]) for row in cursor.fetchall()]

    def _vector_search(
        self,
        query_embedding: np.ndarray,
        limit: int = 20,
        days: int = 30,
    ) -> list[tuple[str, float]]:
        """Search using vector cosine similarity.

        Args:
            query_embedding: Query vector (384-dim)
            limit: Maximum results
            days: Days to look back

        Returns:
            List of (hash, similarity_score) tuples, sorted by score desc
        """
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())

        # Load all embeddings (for small DB this is fine)
        cursor = self.conn.execute(
            """
            SELECT e.hash, e.embedding
            FROM story_embeddings e
            JOIN stories s ON e.hash = s.hash
            WHERE s.processed_at >= ?
            """,
            (cutoff,)
        )

        results = []
        for row in cursor.fetchall():
            embedding = _blob_to_embedding(row["embedding"])
            similarity = _cosine_similarity(query_embedding, embedding)
            results.append((row["hash"], similarity))

        # Sort by similarity descending and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        limit: int = 5,
        days: int = 30,
    ) -> SearchResults:
        """Hybrid search combining BM25 and vector similarity.

        Uses Reciprocal Rank Fusion (RRF) to combine rankings from
        both BM25 (keyword) and vector (semantic) search.

        Args:
            query: Text query for BM25 search
            query_embedding: Optional embedding for vector search
            limit: Maximum results to return
            days: Days to look back

        Returns:
            SearchResults with related stories
        """
        logger.debug("Hybrid search | query='%s' days=%d", query[:50], days)

        rankings = []

        # BM25 search
        try:
            bm25_results = self._bm25_search(query, limit=limit * 2, days=days)
            if bm25_results:
                rankings.append(bm25_results)
                logger.debug("BM25 results | count=%d", len(bm25_results))
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)

        # Vector search (if embedding provided)
        if query_embedding is not None:
            try:
                vector_results = self._vector_search(query_embedding, limit=limit * 2, days=days)
                if vector_results:
                    rankings.append(vector_results)
                    logger.debug("Vector results | count=%d", len(vector_results))
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        if not rankings:
            return SearchResults(query=query, stories=[])

        # Combine rankings with RRF
        combined = _reciprocal_rank_fusion(rankings)

        # Track which source each result came from
        bm25_hashes = {h for h, _ in rankings[0]} if rankings else set()
        vector_hashes = {h for h, _ in rankings[1]} if len(rankings) > 1 else set()

        # Fetch story details for top results
        top_hashes = [h for h, _ in combined[:limit]]
        if not top_hashes:
            return SearchResults(query=query, stories=[])

        placeholders = ",".join("?" * len(top_hashes))
        cursor = self.conn.execute(
            f"""
            SELECT hash, title, pub_date, summary
            FROM stories
            WHERE hash IN ({placeholders})
            """,
            top_hashes
        )

        # Build story objects
        story_map = {}
        for row in cursor.fetchall():
            pub_date = datetime.fromtimestamp(row["pub_date"], timezone.utc).strftime("%Y-%m-%d")
            hash_val = row["hash"]

            # Determine source
            in_bm25 = hash_val in bm25_hashes
            in_vector = hash_val in vector_hashes
            if in_bm25 and in_vector:
                source = "both"
            elif in_vector:
                source = "vector"
            else:
                source = "bm25"

            story_map[hash_val] = RelatedStory(
                hash=hash_val,
                title=row["title"],
                pub_date=pub_date,
                summary=row["summary"] or "",
                score=0.0,  # Will be filled from combined scores
                source=source,
            )

        # Preserve RRF ordering and add scores
        stories = []
        for hash_val, score in combined[:limit]:
            if hash_val in story_map:
                story_map[hash_val].score = score
                stories.append(story_map[hash_val])

        logger.debug("Hybrid search complete | results=%d", len(stories))
        return SearchResults(query=query, stories=stories)

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
            Dictionary with total, important, and embedded story counts
        """
        cursor = self.conn.execute(
            "SELECT COUNT(*) as total, SUM(is_important) as important FROM stories"
        )
        row = cursor.fetchone()

        cursor = self.conn.execute("SELECT COUNT(*) as embedded FROM story_embeddings")
        embedded_row = cursor.fetchone()

        return {
            "total": row["total"] or 0,
            "important": row["important"] or 0,
            "embedded": embedded_row["embedded"] or 0,
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
