"""
Vectorizer Module.

This module provides text embedding generation and chunking functionality
using sentence-transformers models.

Example:
    >>> from app.services.embedding_service import Vectorizer
    >>> service = Vectorizer()
    >>> chunks = service.chunk_text("Long document text...")
    >>> embeddings = service.generate_embeddings(chunks)
"""

import logging
import os
from typing import Optional, Sequence

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


class Vectorizer:
    """
    Class for text embedding generation and chunking.

    This class handles loading the embedding model and provides methods
    for text chunking and embedding generation.

    Attributes:
        model_name (str): Name of the sentence-transformer model.
        chunk_size (int): Size of text chunks in characters.
        chunk_overlap (int): Overlap between consecutive chunks.

    Example:
        >>> vectorizer = Vectorizer()
        >>> chunks = vectorizer.chunk_text("Long medical document...")
        >>> embeddings = vectorizer.generate_embeddings(chunks)
    """

    _model: Optional[SentenceTransformer] = None
    _current_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the Vectorizer.

        Args:
            model_name (str, optional): Sentence-transformer model name.
                Defaults to EMBEDDING_MODEL env var or "paraphrase-multilingual-MiniLM-L12-v2".
            chunk_size (int): Number of characters per chunk. Defaults to 1000.
            chunk_overlap (int): Overlap between chunks. Defaults to 50.
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

        try:
            Vectorizer._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as e:
            logger.error(f"Error loading embedding model '{self.model_name}': {e}")
            raise

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.debug(
            f"Vectorizer initialized: model={self.model_name}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """
        Get or load the embedding model (lazy initialization, singleton).

        The model is shared across all Vectorizer instances to
        avoid loading it multiple times.

        Returns:
            SentenceTransformer: The loaded model instance.
        """
        if (
            Vectorizer._model is None
            or Vectorizer._current_model_name != self.model_name
        ):
            logger.info(f"Loading embedding model: {self.model_name}")

            try:
                Vectorizer._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as e:
                logger.error(f"Error loading embedding model '{self.model_name}': {e}")
                raise

            Vectorizer._current_model_name = self.model_name
            logger.info("Embedding model loaded successfully")

        return Vectorizer._model

    def generate_embeddings(self, texts: list[str]) -> list[Sequence[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (list[str]): List of text strings to embed.

        Returns:
            list[Sequence[float]]: List of embedding vectors.

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

    def embed_single(self, text: str) -> Sequence[float]:
        """
        Generate embedding for a single text.

        Args:
            text (str): The text to embed.

        Returns:
            Sequence[float]: The embedding vector.

        Example:
            >>> embedding = service.embed_single("diabetes treatment")
            >>> len(embedding)
            384
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> Optional[int]:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Optional[int]: The dimension of embedding vectors produced by the model.

        Example:
            >>> dim = service.get_embedding_dimension()
            >>> print(f"Embedding dimension: {dim}")
        """
        return self.model.get_sentence_embedding_dimension()

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunk text into smaller pieces with overlapping segments to maintain context.

        Args:
            text (str): The text to chunk.

        Returns:
            list[str]: List of text chunks.

        Example:
            >>> chunks = service.chunk_text("Long medical document...")
            >>> len(chunks)
            5
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        logger.debug(f"Text chunked into {len(chunks)} pieces")
        return chunks
