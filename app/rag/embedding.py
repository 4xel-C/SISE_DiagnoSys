"""
Embedding Service Module.

This module provides text embedding generation and chunking functionality
using sentence-transformers models.

Example:
    >>> from app.services.embedding_service import EmbeddingService
    >>> service = EmbeddingService()
    >>> chunks = service.chunk_text("Long document text...")
    >>> embeddings = service.generate_embeddings(chunks)
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for text embedding generation and chunking.

    This class handles loading the embedding model and provides methods
    for text chunking and embedding generation.

    Attributes:
        model_name (str): Name of the sentence-transformer model.
        chunk_size (int): Size of text chunks in characters.
        chunk_overlap (int): Overlap between consecutive chunks.

    Example:
        >>> service = EmbeddingService()
        >>> chunks = service.chunk_text("Long medical document...")
        >>> embeddings = service.generate_embeddings(chunks)
    """

    _model: Optional[SentenceTransformer] = None
    _current_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the EmbeddingService.

        Args:
            model_name (str, optional): Sentence-transformer model name.
                Defaults to EMBEDDING_MODEL env var or "all-MiniLM-L6-v2".
            chunk_size (int): Character count per chunk. Defaults to 500.
            chunk_overlap (int): Overlap between chunks. Defaults to 50.
        """
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.debug(
            f"EmbeddingService initialized: model={self.model_name}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """
        Get or load the embedding model (lazy initialization, singleton).

        The model is shared across all EmbeddingService instances to
        avoid loading it multiple times.

        Returns:
            SentenceTransformer: The loaded model instance.
        """
        if (
            EmbeddingService._model is None
            or EmbeddingService._current_model_name != self.model_name
        ):
            logger.info(f"Loading embedding model: {self.model_name}")
            EmbeddingService._model = SentenceTransformer(self.model_name)
            EmbeddingService._current_model_name = self.model_name
            logger.info("Embedding model loaded successfully")

        return EmbeddingService._model

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks of fixed size.

        Args:
            text (str): The text to split into chunks.

        Returns:
            list[str]: List of text chunks.

        Example:
            >>> service = EmbeddingService(chunk_size=100, chunk_overlap=20)
            >>> chunks = service.chunk_text("Long document text...")
            >>> len(chunks)
            5
        """
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())

            start += self.chunk_size - self.chunk_overlap

        logger.debug(f"Text split into {len(chunks)} chunks")
        return chunks

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.

        Example:
            >>> embeddings = service.generate_embeddings(["Hello", "World"])
            >>> len(embeddings)
            2
            >>> len(embeddings[0])  # Dimension of embedding
            384
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False)
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector.

        Example:
            >>> embedding = service.embed_single("diabetes treatment")
            >>> len(embedding)
            384
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    def chunk_and_embed(self, text: str) -> tuple[list[str], list[list[float]]]:
        """
        Chunk text and generate embeddings in one call.

        Args:
            text (str): The text to chunk and embed.

        Returns:
            tuple: A tuple of (chunks, embeddings).

        Example:
            >>> chunks, embeddings = service.chunk_and_embed("Long document...")
            >>> len(chunks) == len(embeddings)
            True
        """
        chunks = self.chunk_text(text)
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            int: The dimension of embedding vectors produced by the model.

        Example:
            >>> dim = service.get_embedding_dimension()
            >>> print(f"Embedding dimension: {dim}")
        """
        return self.model.get_sentence_embedding_dimension()


# Default embedding service instance
embedding_service = EmbeddingService()
