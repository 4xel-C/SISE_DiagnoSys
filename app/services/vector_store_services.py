"""
Vector Service Module.

This module provides CRUD operations for vector embeddings in ChromaDB,
handling documents and patient records for the RAG system.

Example:
    >>> from app.services.vector_service import DocumentVectorService, PatientVectorService
    >>> doc_service = DocumentVectorService()
    >>> doc_service.add("doc_001", "Medical content...", {"type": "protocol"})
    >>> results = doc_service.search("diabetes treatment")
"""

import logging
from typing import Optional

from app.config.vector_db import CollectionType, vector_db
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


# TODO: To be refactored
class BaseVectorService:
    """
    Base class for vector database operations.

    Provides common functionality for adding, searching, and deleting
    vector embeddings in ChromaDB.

    Attributes:
        collection_type (CollectionType): The type of collection to use.
        embedding_service (EmbeddingService): Service for generating embeddings.
    """

    def __init__(
        self,
        collection_type: CollectionType,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize the BaseVectorService.

        Args:
            collection_type (CollectionType): The type of collection to use.
            embedding_service (EmbeddingService, optional): Custom embedding service.
                Defaults to a new EmbeddingService instance.
        """
        self.collection_type = collection_type
        self.embedding_service = embedding_service or EmbeddingService()
        self._id_field = "doc_id"

        logger.debug(f"BaseVectorService initialized for {collection_type.value}")

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
        Add an item to the vector database.

        The content is split into chunks, embedded, and stored with metadata.

        Args:
            item_id (str): Unique identifier for the item.
            content (str): The text content to embed.
            metadata (dict, optional): Additional metadata to store with chunks.

        Returns:
            int: Number of chunks added to the database.
        """
        logger.debug(f"Adding item: {item_id}")

        chunks, embeddings = self.embedding_service.chunk_and_embed(content)
        if not chunks:
            logger.warning(f"Item {item_id} produced no chunks")
            return 0

        chunk_ids = [f"{item_id}_chunk_{i}" for i in range(len(chunks))]

        chunk_metadata = []
        for i in range(len(chunks)):
            meta = {
                self._id_field: item_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            if metadata:
                meta.update(metadata)
            chunk_metadata.append(meta)

        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadata,
        )

        logger.info(f"Item '{item_id}' added with {len(chunks)} chunks")
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
            n_results (int): Maximum number of results to return. Defaults to 5.
            where (dict, optional): Metadata filter for the search.

        Returns:
            list[dict]: List of search results with similarity scores.
        """
        logger.debug(f"Searching for: '{query}' (n_results={n_results})")

        query_embedding = self.embedding_service.embed_single(query)

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
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
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

        existing = self.collection.get(where={self._id_field: item_id})

        if not existing["ids"]:
            logger.debug(f"No chunks found for item {item_id}")
            return False

        self.collection.delete(ids=existing["ids"])
        logger.info(f"Deleted {len(existing['ids'])} chunks for item {item_id}")
        return True

    def get_chunks(self, item_id: str) -> list[dict]:
        """
        Retrieve all chunks for a specific item.

        Args:
            item_id (str): The item ID to retrieve chunks for.

        Returns:
            list[dict]: List of chunks with their metadata, sorted by index.
        """
        results = self.collection.get(
            where={self._id_field: item_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )

        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return chunks

    def count(self) -> int:
        """
        Get the total number of chunks in the collection.

        Returns:
            int: Total chunk count.
        """
        return self.collection.count()

    def exists(self, item_id: str) -> bool:
        """
        Check if an item exists in the collection.

        Args:
            item_id (str): The item ID to check.

        Returns:
            bool: True if the item exists, False otherwise.
        """
        existing = self.collection.get(
            where={self._id_field: item_id},
            limit=1,
        )
        return bool(existing["ids"])


class DocumentVectorService(BaseVectorService):
    """
    Service for document vector operations.

    Handles adding, searching, and managing document embeddings
    in the documents collection.

    Example:
        >>> service = DocumentVectorService()
        >>> service.add_document(
        ...     doc_id="protocol_001",
        ...     content="Medical protocol content...",
        ...     metadata={"type": "protocol", "source": "guidelines.pdf"}
        ... )
        >>> results = service.search("diabetes treatment")
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the DocumentVectorService.

        Args:
            embedding_service (EmbeddingService, optional): Custom embedding service.
        """
        super().__init__(CollectionType.DOCUMENTS, embedding_service)
        self._id_field = "doc_id"

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a document to the vector database.

        Args:
            doc_id (str): Unique identifier for the document.
            content (str): The full document text content.
            metadata (dict, optional): Additional metadata (source, type, title, url).

        Returns:
            int: Number of chunks added to the database.

        Example:
            >>> n_chunks = service.add_document(
            ...     doc_id="protocol_001",
            ...     content="Full medical protocol text...",
            ...     metadata={"source": "guidelines.pdf", "type": "protocol"}
            ... )
        """
        return self.add(doc_id, content, metadata)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector database.

        Args:
            doc_id (str): The document ID to delete.

        Returns:
            bool: True if deleted, False if not found.
        """
        return self.delete(doc_id)

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """
        Retrieve all chunks for a document.

        Args:
            doc_id (str): The document ID.

        Returns:
            list[dict]: List of chunks with metadata.
        """
        return self.get_chunks(doc_id)

    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Search documents with optional type filter.

        Args:
            query (str): The search query.
            n_results (int): Maximum results to return.
            doc_type (str, optional): Filter by document type.

        Returns:
            list[dict]: Search results.

        Example:
            >>> results = service.search_documents(
            ...     "diabetes treatment",
            ...     doc_type="protocol"
            ... )
        """
        where = {"type": doc_type} if doc_type else None
        return self.search(query, n_results, where)


class PatientVectorService(BaseVectorService):
    """
    Service for patient record vector operations.

    Handles adding, searching, and managing patient embeddings
    in the patients collection.

    Example:
        >>> service = PatientVectorService()
        >>> service.add_patient(
        ...     patient_id="patient_001",
        ...     content="Patient symptoms and history...",
        ...     metadata={"nom": "Dupont", "gravite": "rouge"}
        ... )
        >>> results = service.search("chest pain dyspnea")
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize the PatientVectorService.

        Args:
            embedding_service (EmbeddingService, optional): Custom embedding service.
        """
        super().__init__(CollectionType.PATIENTS, embedding_service)
        self._id_field = "patient_id"

    def add_patient(
        self,
        patient_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Add a patient record to the vector database.

        Args:
            patient_id (str): Unique identifier for the patient; meant to be the same as the sqlite db ID.
            content (str): The patient's medical record text.
            metadata (dict, optional): Additional metadata (nom, prenom, gravite).

        Returns:
            int: Number of chunks added to the database.

        Example:
            >>> n_chunks = service.add_patient(
            ...     patient_id="patient_001",
            ...     content="Patient presents with chest pain...",
            ...     metadata={"nom": "Dupont", "prenom": "Jean", "gravite": "rouge"}
            ... )
        """
        if metadata is None:
            metadata = {}
        metadata["source_type"] = "patient"
        return self.add(patient_id, content, metadata)

    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete a patient from the vector database.

        Args:
            patient_id (str): The patient ID to delete.

        Returns:
            bool: True if deleted, False if not found.
        """
        return self.delete(patient_id)

    def get_patient_chunks(self, patient_id: str) -> list[dict]:
        """
        Retrieve all chunks for a patient.

        Args:
            patient_id (str): The patient ID.

        Returns:
            list[dict]: List of chunks with metadata.
        """
        return self.get_chunks(patient_id)

    def search_patients(
        self,
        query: str,
        n_results: int = 5,
        gravite: Optional[str] = None,
    ) -> list[dict]:
        """
        Search patients with optional severity filter.

        Args:
            query (str): The search query.
            n_results (int): Maximum results to return.
            gravite (str, optional): Filter by triage severity.

        Returns:
            list[dict]: Search results.

        Example:
            >>> results = service.search_patients(
            ...     "respiratory distress",
            ...     gravite="rouge"
            ... )
        """
        where = {"gravite": gravite} if gravite else None
        return self.search(query, n_results, where)

    def search_similar_patients(
        self,
        patient_id: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Find patients similar to a given patient.

        Args:
            patient_id (str): The reference patient ID.
            n_results (int): Maximum results to return.

        Returns:
            list[dict]: Similar patients (excluding the reference).

        Example:
            >>> similar = service.search_similar_patients("patient_001")
        """
        chunks = self.get_patient_chunks(patient_id)
        if not chunks:
            return []

        combined_text = " ".join(chunk["text"] for chunk in chunks)

        results = self.search(combined_text, n_results + 1)

        return [r for r in results if r["metadata"].get("patient_id") != patient_id][
            :n_results
        ]


# Default service instances
document_vector_service = DocumentVectorService()
patient_vector_service = PatientVectorService()
