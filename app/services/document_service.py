"""
Document Service Module.

This module provides database query operations for Document records.

Example:
    >>> from app.services import DocumentService
    >>> service = DocumentService()
    >>> documents = service.get_all()
"""

import logging
from typing import Optional

from app.config.database import db
from app.models import Document

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service class for Document database operations.

    Provides CRUD operations and common queries for Document records.

    Example:
        >>> service = DocumentService()
        >>> all_docs = service.get_all()
        >>> doc = service.get_by_id(1)
    """

    def get_all(self) -> list[Document]:
        """
        Retrieve all documents from the database.

        Returns:
            list[Document]: List of all Document records.

        Example:
            >>> documents = service.get_all()
            >>> for doc in documents:
            ...     print(doc.titre)
        """
        logger.debug("Fetching all documents.")
        with db.session() as session:
            documents = session.query(Document).all()
            logger.debug(f"Found {len(documents)} documents.")
            return documents

    def get_by_id(self, document_id: int) -> Optional[Document]:
        """
        Retrieve a document by ID.

        Args:
            document_id (int): The document's unique identifier.

        Returns:
            Optional[Document]: The Document if found, None otherwise.

        Example:
            >>> doc = service.get_by_id(1)
            >>> if doc:
            ...     print(doc.titre)
        """
        logger.debug(f"Fetching document with id={document_id}.")
        with db.session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                logger.debug(f"Found document: {document.titre}")
            else:
                logger.debug(f"Document with id={document_id} not found.")
            return document

    def search_by_titre(self, search_term: str) -> list[Document]:
        """
        Search documents by title (case-insensitive partial match).

        Args:
            search_term (str): The search term to match in document titles.

        Returns:
            list[Document]: List of documents matching the search term.

        Example:
            >>> docs = service.search_by_titre("diabetes")
        """
        logger.debug(f"Searching documents with titre containing '{search_term}'.")
        with db.session() as session:
            documents = (
                session.query(Document)
                .filter(Document.titre.ilike(f"%{search_term}%"))
                .all()
            )
            logger.debug(f"Found {len(documents)} documents matching '{search_term}'.")
            return documents

    def create(self, titre: str, url: str) -> Document:
        """
        Create a new document record.

        Args:
            titre (str): The document title.
            url (str): The source URL of the document.

        Returns:
            Document: The newly created Document record.

        Example:
            >>> doc = service.create(
            ...     titre="Hypertension Guidelines 2024",
            ...     url="https://example.com/hypertension.pdf"
            ... )
        """
        logger.debug(f"Creating new document: {titre}.")
        with db.session() as session:
            document = Document(titre=titre, url=url)
            session.add(document)
            session.commit()
            logger.info(f"Created document: {document}")
            return document

    def delete(self, document_id: int) -> bool:
        """
        Delete a document by ID.

        Args:
            document_id (int): The document's unique identifier.

        Returns:
            bool: True if deleted, False if document not found.

        Example:
            >>> deleted = service.delete(1)
        """
        logger.debug(f"Deleting document with id={document_id}.")
        with db.session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                session.delete(document)
                session.commit()
                logger.info(f"Deleted document with id={document_id}.")
                return True
            logger.debug(f"Document with id={document_id} not found for deletion.")
            return False
