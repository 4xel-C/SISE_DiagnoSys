"""
Vector Database Configuration Module.

This module provides ChromaDB client and collection management for storing
document and patient embeddings used in RAG (Retrieval-Augmented Generation) system.

Example:
    >>> from app.config.vector_db import vector_db, CollectionType
    >>> collection = vector_db.get_collection(CollectionType.DOCUMENTS)
    >>> collection.count()
"""

import logging
import os
from enum import Enum
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Enum for available vector database collections."""

    DOCUMENTS = "medical_documents"
    PATIENTS = "patient_records"


class VectorDatabase:
    """
    ChromaDB client and collection manager.

    This class handles ChromaDB client initialization and collection access.

    Attributes:
        db_path (str): Path to the ChromaDB persistent storage.

    Example:
        >>> db = VectorDatabase()
        >>> docs_collection = db.get_collection(CollectionType.DOCUMENTS)
        >>> patients_collection = db.get_collection(CollectionType.PATIENTS)
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the VectorDatabase manager.

        Args:
            db_path (str, optional): Path to ChromaDB storage.
                Defaults to CHROMA_DB_PATH env var or "data/chroma".
        """
        self.db_path = db_path or os.getenv("CHROMA_DB_PATH", "data/chroma")
        self._client: Optional[ClientAPI] = None
        self._collections: dict[str, chromadb.Collection] = {}

        logger.debug(f"VectorDatabase initialized: path={self.db_path}")

    @property
    def client(self) -> ClientAPI:
        """
        Get or create the ChromaDB client (lazy initialization).

        Returns:
            chromadb.PersistentClient: ChromaDB client instance.
        """
        if self._client is None:
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.debug(f"ChromaDB client created at {self.db_path}")
        return self._client

    def get_collection(self, collection_type: CollectionType) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection by type.

        Args:
            collection_type (CollectionType): The type of collection to get.

        Returns:
            chromadb.Collection: ChromaDB collection instance.

        Example:
            >>> docs = db.get_collection(CollectionType.DOCUMENTS)
            >>> patients = db.get_collection(CollectionType.PATIENTS)
        """
        collection_name = collection_type.value

        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(f"ChromaDB collection '{collection_name}' ready")

        return self._collections[collection_name]

    def delete_collection(self, collection_type: CollectionType) -> bool:
        """
        Delete a ChromaDB collection.

        Args:
            collection_type (CollectionType): The type of collection to delete.

        Returns:
            bool: True if deleted successfully.

        Example:
            >>> db.delete_collection(CollectionType.DOCUMENTS)
        """
        collection_name = collection_type.value

        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
            logger.info(f"Collection '{collection_name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def list_collections(self) -> list[str]:
        """
        List all available collections.

        Returns:
            list[str]: List of collection names.
        """
        return [col.name for col in self.client.list_collections()]

    def get_stats(self) -> dict:
        """
        Get statistics about all collections.

        Returns:
            dict: Statistics for each collection.

        Example:
            >>> stats = db.get_stats()
            >>> print(stats)
        """
        stats = {"db_path": self.db_path, "collections": {}}

        for collection_type in CollectionType:
            collection = self.get_collection(collection_type)
            stats["collections"][collection_type.value] = {
                "count": collection.count(),
            }

        return stats


# Default vector database instance
vector_db = VectorDatabase()
