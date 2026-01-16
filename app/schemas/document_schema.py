"""
Document Schema Module for Vector Database.

This module defines Pydantic schemas for transforming Document ORM models
into data suitable for ChromaDB vector storage.

Example:
    >>> from app.models.document import Document
    >>> from app.schemas.document_schema import DocumentSchema
    >>> doc_orm = session.query(Document).get(1)
    >>> schema = DocumentSchema.model_validate(doc_orm)
    >>> schema.vector_id
    'document_1'
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, computed_field


class DocumentSchema(BaseModel):
    """
    Pydantic schema for document data indexing in ChromaDB.

    This schema transforms SQLAlchemy Document model into a format suitable
    for vector database storage. For documents, chunking is typically required
    before indexing, so this schema also provides chunk-aware methods.

    Attributes:
        id (int): Document primary key from database.
        titre (str): Document title.
        contenu (str): Full document content.
        url (str): Source URL of the document.
        created_at (datetime, optional): Document creation timestamp.

    Example:
        >>> schema = DocumentVectorSchema.model_validate(doc_orm)
        >>> # For full document (small docs)
        >>> collection.add(
        ...     ids=[schema.vector_id],
        ...     documents=[schema.content_for_embedding],
        ...     metadatas=[schema.to_metadata()]
        ... )
        >>> # For chunked document (large docs)
        >>> for chunk_id, chunk_text, metadata in schema.to_chunks(chunks):
        ...     collection.add(ids=[chunk_id], documents=[chunk_text], metadatas=[metadata])
    """

    id: int
    titre: str
    contenu: str
    url: str
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def content_for_embedding(self) -> str:
        """
        Generate text content to be vectorized.

        Combines title and content for better semantic search.

        Returns:
            str: Document title and content for vectorization.
        """
        return f"{self.titre}\n\n{self.contenu}"

    def to_metadata(self, chunk_index: Optional[int] = None) -> dict:
        """
        Generate metadata dictionary for ChromaDB storage.

        Args:
            chunk_index (int, optional): Index of the chunk if document is chunked.

        Returns:
            dict: Metadata with document identifiers and source info.
        """
        metadata = {
            "document_id": self.id,
            "titre": self.titre,
            "url": self.url,
        }

        if self.created_at:
            metadata["created_at"] = self.created_at.isoformat()

        if chunk_index is not None:
            metadata["chunk_index"] = chunk_index

        return metadata

    def chunk_id(self, chunk_index: int) -> str:
        """
        Generate unique identifier for a document chunk.

        Args:
            chunk_index (int): Index of the chunk.

        Returns:
            str: Unique ID in format 'document_{id}_chunk_{index}'.
        """
        return f"document_{self.id}_chunk_{chunk_index}"

    def generate_chunk_metadata(self, chunks: list[str]) -> list[tuple[str, str, dict]]:
        """
        Prepare chunked document data for ChromaDB insertion.

        Args:
            chunks (list[str]): List of text chunks from the document.

        Returns:
            list[tuple[str, str, dict]]: List of (chunk_id, chunk_text, metadata)
                tuples ready for ChromaDB insertion.

        Example:
            >>> chunks = embedding_service.chunk_text(schema.contenu)
            >>> for chunk_id, chunk_text, metadata in schema.to_chunks(chunks):
            ...     collection.add(
            ...         ids=[chunk_id],
            ...         documents=[chunk_text],
            ...         metadatas=[metadata]
            ...     )
        """
        return [
            (
                self.chunk_id(i),
                f"{self.titre}\n\n{chunk}",  # Include title for context
                self.to_metadata(chunk_index=i),
            )
            for i, chunk in enumerate(chunks)
        ]
