"""
Document Service Module.

This module provides database query operations for Document records.

Example:
    >>> from app.services import DocumentService
    >>> service = DocumentService()
    >>> documents = service.get_all()
"""

import logging

from app.config import Database, db
from app.models import Document
from app.rag import document_store
from app.schemas import DocumentSchema

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

    def __init__(self, db_manager: Database = db):
        """
        Initialize the DocumentService.

        Args:
            db_session: Database session/connection. Defaults to app's db.
        """
        self.db_manager = db_manager
        logger.debug("DocumentService initialized.")

    ################################################################
    # READ METHODS
    ################################################################

    def get_all(self) -> list[DocumentSchema]:
        """
        Retrieve all documents from the database.

        Returns:
            list[DocumentSchema]: List of all Document records.

        Example:
            >>> documents = service.get_all()
            >>> for doc in documents:
            ...     print(doc.titre)
        """
        logger.debug("Fetching all documents.")

        results = list()

        with self.db_manager.session() as session:
            documents = session.query(Document).all()
            logger.debug(f"Found {len(documents)} documents.")

            for document in documents:
                results.append(DocumentSchema.model_validate(document))

        return results

    def get_by_id(self, document_id: int) -> DocumentSchema:
        """
        Retrieve a document by ID.

        Args:
            document_id (int): The document's unique identifier.

        Returns:
            Optional[DocumentSchema]: The Document if found, None otherwise.

        Example:
            >>> doc = service.get_by_id(1)
            >>> if doc:
            ...     print(doc.titre)
        """
        logger.debug(f"Fetching document with id={document_id}.")
        with self.db_manager.session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                logger.debug(f"Found document: {document.titre}")
            else:
                logger.debug(f"Document with id={document_id} not found.")

            document_schema = (
                DocumentSchema.model_validate(document) if document else None
            )

            if not document_schema:
                raise ValueError(f"Document with id={document_id} not found.")

            return document_schema

    def search_by_titre(self, search_term: str) -> list[DocumentSchema]:
        """
        Search documents by title (case-insensitive partial match).

        Args:
            search_term (str): The search term to match in document titles.

        Returns:
            list[DocumentSchema]: List of documents matching the search term.

        Example:
            >>> docs = service.search_by_titre("diabetes")
        """
        logger.debug(f"Searching documents with titre containing '{search_term}'.")

        result = list()

        with self.db_manager.session() as session:
            documents = (
                session.query(Document)
                .filter(Document.titre.ilike(f"%{search_term}%"))
                .all()
            )
            logger.debug(f"Found {len(documents)} documents matching '{search_term}'.")

            for document in documents:
                result.append(DocumentSchema.model_validate(document))

        return result

    ################################################################
    # CREATE METHODS
    ################################################################

    def create(self, titre: str, contenu: str, url: str) -> DocumentSchema:
        """
        Create a new document record.

        Args:
            titre (str): The document title.
            contenu (str): The document content.
            url (str): The source URL of the document.

        Returns:
            DocumentSchema: The newly created Document record.

        Example:
            >>> doc = service.create(
            ...     titre="Hypertension Guidelines 2024",
            ...     contenu="Content of the hypertension guidelines.",
            ...     url="https://example.com/hypertension.pdf"
            ... )
        """
        logger.debug(f"Creating new document: {titre}.")
        with self.db_manager.session() as session:
            document = Document(titre=titre, contenu=contenu, url=url)
            session.add(document)
            session.commit()
            logger.info(f"Created document: {document}")

            # add to vector store
            document_store.add(
                item_id=document.vector_id,
                content=document.content_for_embedding,
                metadata=document.to_metadata,
            )

            return DocumentSchema.model_validate(document)

    ################################################################
    # UPDATE METHODS
    ################################################################

    def update_document(
        self, document_id: int, titre: str, contenu: str, url: str
    ) -> DocumentSchema:
        """Update an existing document with new information.

        Args:
            document_id (int): The unique identifier of the document to update.
            titre (str): The new title for the document.
            contenu (str): The new content for the document.
            url (str): The new URL for the document.

        Returns:
            DocumentSchema: The updated Document record.
        """
        logger.debug(f"Updating document with id={document_id}.")
        with self.db_manager.session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                logger.error(f"Document with id={document_id} not found for update.")
                raise ValueError(f"Document with id={document_id} not found.")

            document.titre = titre  # type: ignore
            document.contenu = contenu  # type: ignore
            document.url = url  # type: ignore
            session.commit()
            logger.info(f"Updated document: {document}")

            # update in chroma_db
            document_store.add(
                item_id=document.vector_id,
                content=document.content_for_embedding,
                metadata=document.to_metadata,
            )

            return DocumentSchema.model_validate(document)

    ################################################################
    # DELETE METHODS
    ################################################################
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
        with self.db_manager.session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                session.delete(document)
                session.commit()
                logger.info(f"Deleted document with id={document_id}.")

                # delete from chroma_db
                document_store.delete(item_id=document.vector_id)
                return True
            logger.debug(f"Document with id={document_id} not found for deletion.")
            return False
