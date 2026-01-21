"""Memory infrastructure for semantic story search.

This package provides optional semantic search capabilities using ChromaDB.

VectorStore:
    ChromaDB-based store for story embeddings.
    Enables semantic similarity search across story history.

StoryEmbedding:
    Data class representing a story with its vector embedding.

Requirements:
    pip install chromadb

Enable via configuration:
    ENABLE_MEMORY=true

Example:
    >>> from memory import VectorStore
    >>> store = VectorStore("./vectors")
    >>> await store.add_story(hash, title, summary, pub_date, source_url)
    >>> results = await store.search("AI safety research")
"""

from memory.vector_store import VectorStore, StoryEmbedding

__all__ = [
    "VectorStore",
    "StoryEmbedding",
]
