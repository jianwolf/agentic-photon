"""Vector store for semantic story search using ChromaDB.

This module provides optional semantic search capabilities for stories.
It uses ChromaDB to store embeddings and find similar stories.

Features:
    - Lazy initialization (ChromaDB loaded only when needed)
    - Semantic similarity search across story history
    - Related story discovery by embedding similarity
    - Persistent storage using DuckDB backend

Requirements:
    pip install chromadb

Enable via configuration:
    ENABLE_MEMORY=true
    VECTOR_DB_PATH=./vectors

The vector store complements the keyword-based database search
by finding conceptually related stories even without matching keywords.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StoryEmbedding:
    """A story with its embedding for semantic search.

    Attributes:
        hash: Story deduplication hash
        title: Story headline
        summary: Full summary text (used for embedding)
        pub_date: Publication timestamp
        source_url: RSS feed URL
        embedding: Vector embedding (populated by ChromaDB)
    """
    hash: str
    title: str
    summary: str
    pub_date: datetime
    source_url: str
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """Result from a semantic search.

    Attributes:
        story: The matched story
        distance: Raw distance from query (lower = more similar)
        relevance_score: Normalized score 0-1 (higher = more relevant)
    """
    story: StoryEmbedding
    distance: float
    relevance_score: float


class VectorStore:
    """ChromaDB-based vector store for semantic story search.

    This provides semantic similarity search for stories,
    enabling cross-story context and related story discovery.
    """

    def __init__(self, path: Path | str, collection_name: str = "stories"):
        """Initialize the vector store.

        Args:
            path: Directory path for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.path = Path(path)
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazily initialize ChromaDB.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            return True

        try:
            import chromadb
            from chromadb.config import Settings

            self.path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.path),
                anonymized_telemetry=False
            ))

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Photon news stories"}
            )

            self._initialized = True
            logger.info(f"Vector store initialized at {self.path}")
            return True

        except ImportError:
            logger.warning("ChromaDB not installed. Vector search disabled.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False

    async def add_story(
        self,
        story_hash: str,
        title: str,
        summary: str,
        pub_date: datetime,
        source_url: str,
    ) -> bool:
        """Add a story to the vector store.

        Args:
            story_hash: Unique story identifier
            title: Story title
            summary: Story summary (used for embedding)
            pub_date: Publication date
            source_url: Source URL

        Returns:
            True if added successfully
        """
        if not self._ensure_initialized():
            return False

        try:
            # Combine title and summary for embedding
            text = f"{title}\n\n{summary}"

            self._collection.add(
                ids=[story_hash],
                documents=[text],
                metadatas=[{
                    "title": title,
                    "pub_date": pub_date.isoformat(),
                    "source_url": source_url,
                }]
            )

            logger.debug(f"Added story to vector store: {story_hash}")
            return True

        except Exception as e:
            logger.error(f"Failed to add story to vector store: {e}")
            return False

    async def search(
        self,
        query: str,
        n_results: int = 5,
        min_relevance: float = 0.5,
    ) -> list[SearchResult]:
        """Search for semantically similar stories.

        Args:
            query: Search query text
            n_results: Maximum number of results
            min_relevance: Minimum relevance score (0-1)

        Returns:
            List of search results sorted by relevance
        """
        if not self._ensure_initialized():
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
            )

            search_results = []
            for i, (id, doc, metadata, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # Convert distance to relevance score (0-1)
                relevance = 1.0 / (1.0 + distance)

                if relevance < min_relevance:
                    continue

                story = StoryEmbedding(
                    hash=id,
                    title=metadata.get("title", ""),
                    summary=doc,
                    pub_date=datetime.fromisoformat(metadata.get("pub_date", "")),
                    source_url=metadata.get("source_url", ""),
                )

                search_results.append(SearchResult(
                    story=story,
                    distance=distance,
                    relevance_score=relevance,
                ))

            return search_results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def find_related(
        self,
        story_hash: str,
        n_results: int = 5,
    ) -> list[SearchResult]:
        """Find stories related to a given story.

        Args:
            story_hash: Hash of the story to find related stories for
            n_results: Maximum number of results

        Returns:
            List of related stories (excluding the input story)
        """
        if not self._ensure_initialized():
            return []

        try:
            # Get the story's embedding
            result = self._collection.get(
                ids=[story_hash],
                include=["documents"]
            )

            if not result["documents"]:
                return []

            # Search for similar stories
            doc = result["documents"][0]
            results = await self.search(doc, n_results=n_results + 1)

            # Filter out the original story
            return [r for r in results if r.story.hash != story_hash][:n_results]

        except Exception as e:
            logger.error(f"Find related error: {e}")
            return []

    def count(self) -> int:
        """Get the number of stories in the store.

        Returns:
            Number of stories
        """
        if not self._ensure_initialized():
            return 0

        try:
            return self._collection.count()
        except Exception:
            return 0

    def persist(self) -> None:
        """Persist the vector store to disk."""
        if self._client and self._initialized:
            try:
                self._client.persist()
                logger.debug("Vector store persisted")
            except Exception as e:
                logger.error(f"Failed to persist vector store: {e}")
