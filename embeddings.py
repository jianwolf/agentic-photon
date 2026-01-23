"""Embedding model management for semantic search.

This module provides text embeddings using the BGE-small-en-v1.5 model
via sentence-transformers. Embeddings are used for the vector component
of hybrid RAG search.

Model: BAAI/bge-small-en-v1.5
    - 384 dimensions
    - Fast inference on CPU
    - Good quality for retrieval tasks

Usage:
    >>> from embeddings import get_embeddings
    >>> embeddings = get_embeddings()
    >>> vector = embeddings.encode("OpenAI releases GPT-5")
    >>> vectors = embeddings.encode_batch(["text1", "text2"])
"""

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model.

    Provides lazy loading and caching of the model instance.
    The model is loaded on first use and reused for subsequent calls.

    Attributes:
        model_name: HuggingFace model identifier
        dim: Embedding dimension size
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """Initialize embedding model wrapper.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        self._model = None

    def _load_model(self):
        """Lazily load the sentence-transformers model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded | dim=%d", self.dim)
        return self._model

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to embedding vector.

        Args:
            text: Input text to encode

        Returns:
            numpy array of shape (384,) with float32 values
        """
        model = self._load_model()
        # BGE models recommend adding instruction prefix for queries
        # but for simplicity we skip it here (works well without)
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (len(texts), 384) with float32 values
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dim)

        model = self._load_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)


# Global singleton instance
_embedding_model: EmbeddingModel | None = None


def get_embeddings() -> EmbeddingModel:
    """Get the global embedding model instance.

    Returns a singleton instance that is lazily initialized on first call.
    The model is cached for the lifetime of the process.

    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def encode_text(text: str) -> np.ndarray:
    """Convenience function to encode a single text.

    Args:
        text: Input text to encode

    Returns:
        numpy array of shape (384,)
    """
    return get_embeddings().encode(text)


def encode_texts(texts: list[str]) -> np.ndarray:
    """Convenience function to encode multiple texts.

    Args:
        texts: List of input texts

    Returns:
        numpy array of shape (len(texts), 384)
    """
    return get_embeddings().encode_batch(texts)
