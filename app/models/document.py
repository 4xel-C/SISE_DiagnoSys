"""
Document Model Module.

This module defines the SQLAlchemy ORM model for scientific document records.

Example:
    >>> from app.models.document import Document
    >>> from app.config.database import db
    >>> with db.session() as session:
    ...     doc = Document(
    ...         titre="Diabetes Treatment Guidelines",
    ...         url="https://example.com/diabetes-guidelines.pdf"
    ...     )
    ...     session.add(doc)
    ...     session.commit()
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models import Base


class Document(Base):
    """
    SQLAlchemy model representing a scientific document record.

    This model stores references to scientific documents, medical guidelines,
    and research papers used for the RAG system.

    Attributes:
        id (int): Primary key, auto-incremented unique identifier.
        titre (str): Document title (max 500 characters). Required.
        contenu (str): Content of the document. Required.
        created_at (datetime): Timestamp of record creation. Auto-generated.

    Example:
        >>> doc = Document(
        ...     titre="COVID-19 Clinical Management Guidelines",
        ...     url="https://who.int/covid-guidelines.pdf"
        ... )
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    titre = Column(String(500), nullable=False)
    contenu = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # Relationship to patients via association object (with similarity scores)
    patients_concernes_assoc = relationship(
        "DocumentProche",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        """Return a string representation of the Document instance."""
        return f"<Document(id={self.id}, titre='{self.titre[:50]}...')>"
