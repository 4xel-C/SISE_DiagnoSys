"""
Vector Store Module.

This module provides CRUD operations for vector embeddings in ChromaDB,
handling documents and patient records for the RAG system.

Example:
    >>> from app.rag.vector_store import VectorStore, CollectionType
    >>> store = VectorStore(CollectionType.DOCUMENTS)
    >>> store.add("doc_001", "Medical content...", {"type": "protocol"})
    >>> results = store.search("diabetes treatment")
"""

import logging
from typing import Optional

from app.config.vector_db import CollectionType, vector_db
from app.rag.vectorizer import Vectorizer

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for ChromaDB operations.

    Provides methods for adding, searching, and deleting vector embeddings.
    Works with any collection type (documents, patients).

    Attributes:
        collection_type (CollectionType): The type of collection to use.
        vectorizer (Vectorizer): Service for generating embeddings.
        id_field (str): The metadata field name for item IDs.

    Example:
        >>> store = VectorStore(CollectionType.DOCUMENTS)
        >>> store.add("doc_001", "Medical content...", {"type": "protocol"})
        >>> results = store.search("diabetes treatment", n_results=5)
    """

    def __init__(
        self,
        collection_type: CollectionType,
        vectorizer: Optional[Vectorizer] = None,
        id_field: str = "item_id",
    ):
        """
        Initialize the VectorStore.

        Args:
            collection_type (CollectionType): The collection to use.
            vectorizer (Vectorizer, optional): Custom vectorizer instance.
            id_field (str): Metadata field name for item IDs. Defaults to "item_id".
        """
        self.collection_type = collection_type
        self.vectorizer = vectorizer or Vectorizer()
        self.id_field = id_field

        logger.debug(f"VectorStore initialized for {collection_type.value}")

    @property
    def collection(self):
        """Get the ChromaDB collection."""
        return vector_db.get_collection(self.collection_type)

    def add(
        self,
        item_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add an item to the vector store.

        Chunks the content, generates embeddings, and stores them in ChromaDB.

        Args:
            item_id (str): Unique identifier for the item.
            content (str): The text content to embed.
            metadata (dict, optional): Additional metadata to store with each chunk.

        Returns:
            int: Number of chunks added.

        Example:
            >>> n_chunks = store.add(
            ...     item_id="doc_001",
            ...     content="Long medical document...",
            ...     metadata={"type": "protocol"}
            ... )
        """
        logger.debug(f"Adding item: {item_id}")

        # Delete existing chunks if updating
        self.delete(item_id)

        # Chunk the content
        chunks = self.vectorizer.chunk_text(content)
        if not chunks:
            logger.warning(f"No chunks generated for item {item_id}")
            return 0

        # Generate embeddings
        embeddings = self.vectorizer.generate_embeddings(chunks)

        # Prepare metadata for each chunk
        base_metadata = metadata or {}
        base_metadata[self.id_field] = item_id

        ids = []
        documents = []
        metadatas = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{item_id}_chunk_{i}"
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(chunks)} chunks for item {item_id}")
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar items using semantic search.

        Args:
            query (str): The search query text.
            n_results (int): Maximum number of results. Defaults to 5.
            where (dict, optional): Metadata filter for the search.

        Returns:
            list[dict]: List of results with keys:
                - chunk_id: The chunk identifier
                - text: The chunk content
                - metadata: Associated metadata
                - distance: Distance score (lower = more similar)
                - similarity: Similarity score (higher = more similar)

        Example:
            >>> results = store.search("chest pain", n_results=3)
            >>> results = store.search("diabetes", where={"type": "protocol"})
        """
        logger.debug(f"Searching: '{query}' (n_results={n_results})")

        query_embedding = self.vectorizer.embed_single(query)

        search_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            search_params["where"] = where

        results = self.collection.query(**search_params)

        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                formatted_results.append(
                    {
                        "chunk_id": chunk_id,
                        "text": results["documents"][0][i]
                        if results["documents"]
                        else "",
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": distance,
                        "similarity": similarity,
                    }
                )

        logger.debug(f"Found {len(formatted_results)} results")
        return formatted_results

    def delete(self, item_id: str) -> bool:
        """
        Delete all chunks associated with an item.

        Args:
            item_id (str): The item ID to delete.

        Returns:
            bool: True if chunks were deleted, False if none found.
        """
        logger.debug(f"Deleting item: {item_id}")

        existing = self.collection.get(where={self.id_field: item_id})

        if not existing["ids"]:
            logger.debug(f"No chunks found for item {item_id}")
            return False

        self.collection.delete(ids=existing["ids"])
        logger.info(f"Deleted {len(existing['ids'])} chunks for item {item_id}")
        return True

    def get(self, item_id: str) -> list[dict]:
        """
        Retrieve all chunks for a specific item.

        Args:
            item_id (str): The item ID to retrieve.

        Returns:
            list[dict]: List of chunks sorted by index.
        """
        results = self.collection.get(
            where={self.id_field: item_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i]
                        if results["metadatas"]
                        else {},
                    }
                )

        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return chunks

    def exists(self, item_id: str) -> bool:
        """
        Check if an item exists in the collection.

        Args:
            item_id (str): The item ID to check.

        Returns:
            bool: True if the item exists.
        """
        existing = self.collection.get(
            where={self.id_field: item_id},
            limit=1,
        )
        return bool(existing["ids"])

    def count(self) -> int:
        """
        Get the total number of chunks in the collection.

        Returns:
            int: Total chunk count.
        """
        return self.collection.count()


# Pre-configured store instances
document_store = VectorStore(
    collection_type=CollectionType.DOCUMENTS,
    id_field="doc_id",
)

patient_store = VectorStore(
    collection_type=CollectionType.PATIENTS,
    id_field="patient_id",
)
