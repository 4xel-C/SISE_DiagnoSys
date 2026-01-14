"""
Database Configuration Module.

This module provides database initialization and session management
for the SQLAlchemy ORM layer using a context manager pattern.

Example:
    >>> from app.config.database import Database
    >>> db = Database()
    >>> db.init_db()
    >>> with db.session() as session:
    ...     patient = Patient(nom="Dupont", gravite="jaune")
    ...     session.add(patient)
    ...     session.commit()
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, Document, Patient  # noqa: F401

load_dotenv()


logger = logging.getLogger(__name__)


class Database:
    """
    Database manager class with context manager support for sessions.

    This class provides a singleton-like pattern for database connections
    and a context manager for safe session handling with automatic
    cleanup on exceptions.

    Attributes:
        db_path (str): Path to the SQLite database file.

    Example:
        >>> db = Database()
        >>> db.init_db()
        >>> with db.session() as session:
        ...     patients = session.query(Patient).all()
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the Database manager.

        Args:
            db_path (str, optional): Path to the SQLite database file.
                If not provided, uses DATABASE_PATH env var or defaults
                to "diagnosys.db".
        """
        self.db_path = db_path or os.getenv("DATABASE_PATH", "diagnosys.db")
        logger.debug(f"Initialisation: Database path set to: {self.db_path}")
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None

    @property
    def engine(self) -> Engine:
        """
        Get or create the SQLAlchemy engine (lazy initialization).

        Returns:
            Engine: SQLAlchemy Engine instance connected to the database.
        """
        if self._engine is None:
            self._engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
            logger.debug("SQLAlchemy engine created.")
        return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Provides a transactional scope around a series of operations.
        Automatically handles commit on success and rollback on exception.

        Yields:
            Session: SQLAlchemy Session instance.

        Raises:
            Exception: Re-raises any exception after rollback.

        Example:
            >>> with db.session() as session:
            ...     patient = Patient(nom="Martin", gravite="rouge")
            ...     session.add(patient)
            ...     session.commit()
        """
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)

        session = self._session_factory()
        try:
            yield session
        except Exception:
            logger.exception(
                "Exception occurred during database session, rolling back."
            )
            session.rollback()
            raise
        finally:
            logger.debug("Closing database session.")
            session.close()

    def init_db(self):
        """
        Initialize the database and create all tables.

        This method should be called once at application startup
        to ensure all tables are created.

        Example:
            >>> db = Database()
            >>> db.init_db()
        """
        logger.info("Initializing database and creating tables if not exist.")
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized successfully.")


# Default database instance
db = Database()
