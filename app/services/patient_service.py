"""
Patient Service Module.

This module provides database query operations for Patient records.
All CRUD operations and queries related to Patient are encapsulated in this service class.

Example:
    >>> from app.services import PatientService
    >>> service = PatientService()
    >>> patients = service.get_all()
"""

import logging

from sqlalchemy import and_, or_

from app.config import Database, db
from app.models import Document, Patient
from app.models.patient import DocumentProche, PatientProche
from app.rag import (
    patient_store,
)
from app.schemas import DocumentProcheSchema, PatientProcheSchema, PatientSchema

logger = logging.getLogger(__name__)


class PatientService:
    """
    Service class for Patient database operations.

    Provides CRUD operations and common queries for Patient records.

    Example:
        >>> service = PatientService()
        >>> all_patients = service.get_all()
        >>> critical = service.get_by_gravite("rouge")
    """

    def __init__(self, db_manager: Database = db):
        """
        Initialize the PatientService.

        Args:
            db_manager (Database): Database manager instance. Defaults to global db.
        """
        self.db_manager = db_manager

    ################################################################
    # GET METHODS
    ################################################################

    def get_all(self) -> list[PatientSchema]:
        """
        Retrieve all patients from the database.

        Returns:
            list[PatientSchema]: List of all Patient records.
        Example:
            >>> patients = service.get_all()
            >>> for p in patients:
            ...     print(p.nom)
        """
        logger.debug("Fetching all patients.")

        result = list()

        with self.db_manager.session() as session:
            patients = session.query(Patient).all()
            logger.debug(f"Found {len(patients)} patients.")

            # transform to schema (pydantic python object)
            for patient in patients:
                result.append(PatientSchema.model_validate(patient))

        return result

    def get_by_id(self, patient_id: int) -> PatientSchema:
        """
        Retrieve a patient by ID.

        Args:
            patient_id (int): The patient's unique identifier.

        Returns:
            Optional[PatientSchema]: The Patient if found, None otherwise.

        Example:
            >>> patient = service.get_by_id(1)
            >>> if patient:
            ...     print(patient.nom)
        """
        logger.debug(f"Fetching patient with id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if patient:
                logger.debug(f"Found patient: {patient.nom}")
            else:
                logger.debug(f"Patient with id={patient_id} not found.")

        patient_schema = PatientSchema.model_validate(patient) if patient else None

        if not patient_schema:
            raise ValueError(f"Patient with id={patient_id} not found.")

        return patient_schema

    def get_by_gravite(self, gravite: str) -> list[PatientSchema]:
        """
        Retrieve all patients with a specific severity level.

        Args:
            gravite (str): Severity level ('gris', 'vert', 'jaune', 'rouge').

        Returns:
            list[PatientSchema]: List of patients with the specified severity.

        Example:
            >>> critical = service.get_by_gravite("rouge")
        """
        logger.debug(f"Fetching patients with gravite={gravite}.")
        result = list()
        with self.db_manager.session() as session:
            patients = session.query(Patient).filter_by(gravite=gravite).all()
            logger.debug(f"Found {len(patients)} patients with gravite={gravite}.")

            for patient in patients:
                logger.debug(f"Patient: {patient.nom}, Gravite: {patient.gravite}")
                result.append(PatientSchema.model_validate(patient))

        return result

    def get_by_query(self, query: str) -> list[PatientSchema]:
        """
        Retrieve all patients with names matching a search query.

        Args:
            query (str): Name of patient so search.

        Returns:
            list[PatientSchema]: List of patients matching the query.

        Example:
            >>> result = service.get_by_query("michel dupont")
        """
        logger.debug(f"Fetching patients matching query '{query}'.")
        result = list()
        with self.db_manager.session() as session:
            tokens = query.split()
            conds = [
                or_(Patient.nom.ilike(f"%{t}%"), Patient.prenom.ilike(f"%{t}%"))
                for t in tokens
            ]

            patients = session.query(Patient).filter(and_(*conds)).all()
            logger.debug(f"Found {len(patients)} patients matching query '{query}'.")

            for patient in patients:
                logger.debug(f"Patient: {patient.nom} {patient.prenom}")
                result.append(PatientSchema.model_validate(patient))

        return result

    def get_context(self, patient_id: int) -> str:
        """
        Retrieve the context field of a patient.

        Args:
            patient_id (int): The patient's unique identifier.

        Returns:
            str: The context of the patient.

        Example:
            >>> context = service.get_context(1)
        """
        logger.debug(f"Fetching context for patient id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                logger.error(
                    f"Patient with id={patient_id} not found for context retrieval."
                )
                raise ValueError(f"Patient with id={patient_id} not found.")

            logger.debug(f"Retrieved context for patient id={patient_id}.")
            return patient.contexte if patient.contexte else ""  # type: ignore

    def get_documents_proches(self, patient_id: int) -> list[DocumentProcheSchema]:
        """
        Retrieve related documents for a patient, sorted by similarity score.

        Args:
            patient_id (int): The patient's unique identifier.

        Returns:
            list[DocumentProcheSchema]: List of related documents with scores.

        Raises:
            ValueError: If the patient is not found.

        Example:
            >>> docs = service.get_documents_proches(1)
            >>> for doc in docs:
            ...     print(f"Document {doc.document_id}: {doc.similarity_score}")
        """
        logger.debug(f"Fetching related documents for patient id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                raise ValueError(f"Patient with id={patient_id} not found.")

            associations = (
                session.query(DocumentProche)
                .filter_by(patient_id=patient_id)
                .order_by(DocumentProche.similarity_score.desc())
                .all()
            )

            result = [
                DocumentProcheSchema(
                    document_id=a.document_id,  # type: ignore
                    similarity_score=a.similarity_score,  # type: ignore
                )
                for a in associations
            ]
            logger.debug(
                f"Found {len(result)} related documents for patient id={patient_id}."
            )

        return result

    def get_patients_proches(self, patient_id: int) -> list[PatientProcheSchema]:
        """
        Retrieve related (similar) patients for a patient, sorted by similarity score.

        Args:
            patient_id (int): The patient's unique identifier.

        Returns:
            list[PatientProcheSchema]: List of related patients with scores.

        Raises:
            ValueError: If the patient is not found.

        Example:
            >>> patients = service.get_patients_proches(1)
            >>> for p in patients:
            ...     print(f"Patient {p.patient_id}: {p.similarity_score}")
        """
        logger.debug(f"Fetching related patients for patient id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                raise ValueError(f"Patient with id={patient_id} not found.")

            associations = (
                session.query(PatientProche)
                .filter_by(patient_id=patient_id)
                .order_by(PatientProche.similarity_score.desc())
                .all()
            )

            result = [
                PatientProcheSchema(
                    patient_id=a.patient_proche_id,  # type: ignore
                    similarity_score=a.similarity_score,  # type: ignore
                )
                for a in associations
            ]
            logger.debug(
                f"Found {len(result)} related patients for patient id={patient_id}."
            )

        return result

    ################################################################
    # CREATE METHODS
    ################################################################

    def create(self, **kwargs) -> PatientSchema:
        """
        Create a new patient record.
        With respect to the Patient model fields.

        Args:
            **kwargs: Patient attributes (nom, gravite, type_maladie, etc.)

        Returns:
            PatientSchema: The newly created Patient record.

        Raises:
            ValueError: If required fields are missing.

        Example:
            >>> patient = service.create(
            ...     nom="Dupont",
            ...     gravite="jaune",
            ...     fc=80,
            ...     temperature=37.5
            ... )
        """
        logger.debug(f"Creating new patient with data: {kwargs}.")
        patient_schema: PatientSchema
        with self.db_manager.session() as session:
            patient = Patient(**kwargs)
            session.add(patient)
            session.commit()
            logger.info(f"Created patient: {patient}")
            patient_schema = PatientSchema.model_validate(patient)

        # update chromadb with new patient
        patient_store.add(
            item_id=patient_schema.vector_id,
            content=patient_schema.content_for_embedding,
            metadata=patient_schema.to_metadata(),
            no_chunking=True,
        )

        return patient_schema

    ################################################################
    # DELETE METHODS
    ################################################################

    def delete(self, patient_id: int) -> bool:
        """
        Delete a patient by ID.

        Args:
            patient_id (int): The patient's unique identifier.

        Returns:
            bool: True if deleted, False if patient not found.

        Example:
            >>> deleted = service.delete(1)
        """
        logger.debug(f"Deleting patient with id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if patient:
                session.delete(patient)
                session.commit()
                logger.info(f"Deleted patient with id={patient_id}.")

                # Deletee from chromadb
                patient_store.delete(item_id=patient.vector_id)

                return True
            logger.debug(f"Patient with id={patient_id} not found for deletion.")
            return False

    ################################################################
    # UPDATE METHODS
    ################################################################
    def update_context(self, patient_id: int, new_context: str) -> PatientSchema:
        """
        Update the context field of a patient and update embdeddings from the vector store.

        Args:
            patient_id (int): The patient's unique identifier.
            new_context (str): The new context to set.

        Returns:
            PatientSchema: The updated Patient record.

        Raises:
            ValueError: If the patient is not found.

        Example:
            >>> updated_patient = service.update_context(1, "New context data")
        """
        logger.debug(f"Updating context for patient id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                logger.error(
                    f"Patient with id={patient_id} not found for context update."
                )
                raise ValueError(f"Patient with id={patient_id} not found.")

            patient.contexte = new_context  # type: ignore
            session.commit()
            logger.info(f"Updated context for patient id={patient_id}.")
            patient = PatientSchema.model_validate(patient)

        # Update the chromadb
        patient_store.add(
            item_id=patient.vector_id,
            content=patient.content_for_embedding,
            metadata=patient.to_metadata(),
            no_chunking=True,
        )

        return patient

    def update_documents(
        self,
        patient_id: int,
        document_ids: list[int],
        similarity_scores: list[float] | None = None,
    ) -> PatientSchema:
        """
        Update the related documents of a patient with optional similarity scores.

        Args:
            patient_id (int): The patient's unique identifier.
            document_ids (list[int]): List of document IDs to relate.
            similarity_scores (list[float] | None): Optional list of similarity scores
                corresponding to each document. Must have same length as document_ids.

        Returns:
            PatientSchema: The updated Patient record.

        Raises:
            ValueError: If the patient is not found or if similarity_scores length
                doesn't match document_ids length.

        Example:
            >>> updated_patient = service.update_documents(1, [10, 20], [0.95, 0.87])
        """
        logger.debug(f"Updating documents for patient id={patient_id}.")

        if similarity_scores and len(similarity_scores) != len(document_ids):
            raise ValueError(
                "similarity_scores must have the same length as document_ids."
            )

        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                logger.error(
                    f"Patient with id={patient_id} not found for document update."
                )
                raise ValueError(f"Patient with id={patient_id} not found.")

            # Clear existing relations
            session.query(DocumentProche).filter_by(patient_id=patient_id).delete()

            # Add new relations with scores
            for i, doc_id in enumerate(document_ids):
                document = session.query(Document).filter_by(id=doc_id).first()
                if document:
                    score = similarity_scores[i] if similarity_scores else None
                    assoc = DocumentProche(
                        patient_id=patient_id,
                        document_id=doc_id,
                        similarity_score=score,
                    )
                    session.add(assoc)

            session.commit()
            logger.info(f"Updated documents for patient id={patient_id}.")
            updated_patient = PatientSchema.model_validate(patient)

        return updated_patient

    def update_close_patients(
        self,
        patient_id: int,
        close_patient_ids: list[int],
        similarity_scores: list[float] | None = None,
    ) -> PatientSchema:
        """
        Update the related close patients of a patient with optional similarity scores.

        Args:
            patient_id (int): The patient's unique identifier.
            close_patient_ids (list[int]): List of close patient IDs to relate.
            similarity_scores (list[float] | None): Optional list of similarity scores
                corresponding to each close patient. Must have same length as close_patient_ids.

        Returns:
            PatientSchema: The updated Patient record.

        Raises:
            ValueError: If the patient is not found or if similarity_scores length
                doesn't match close_patient_ids length.

        Example:
            >>> updated_patient = service.update_close_patients(1, [2, 3], [0.92, 0.85])
        """
        logger.debug(f"Updating close patients for patient id={patient_id}.")

        if similarity_scores and len(similarity_scores) != len(close_patient_ids):
            raise ValueError(
                "similarity_scores must have the same length as close_patient_ids."
            )

        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                logger.error(
                    f"Patient with id={patient_id} not found for close patients update."
                )
                raise ValueError(f"Patient with id={patient_id} not found.")

            # Clear existing relations
            session.query(PatientProche).filter_by(patient_id=patient_id).delete()

            # Add new relations with scores
            for i, close_id in enumerate(close_patient_ids):
                close_patient = session.query(Patient).filter_by(id=close_id).first()
                if close_patient:
                    score = similarity_scores[i] if similarity_scores else None
                    assoc = PatientProche(
                        patient_id=patient_id,
                        patient_proche_id=close_id,
                        similarity_score=score,
                    )
                    session.add(assoc)

            session.commit()
            logger.info(f"Updated close patients for patient id={patient_id}.")
            updated_patient = PatientSchema.model_validate(patient)

        return updated_patient

    def update_diagnosys(self, patient_id: int, new_diagnosys: str) -> PatientSchema:
        """
        Update the diagnosys field of a patient.

        Args:
            patient_id (int): The patient's unique identifier.
            new_diagnosys (str): The new diagnosys to set.

        Returns:
            PatientSchema: The updated Patient record.

        Raises:
            ValueError: If the patient is not found.

        Example:
            >>> updated_patient = service.update_diagnosys(1, "New diagnosys data")
        """
        logger.debug(f"Updating diagnosys for patient id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                logger.error(
                    f"Patient with id={patient_id} not found for diagnosys update."
                )
                raise ValueError(f"Patient with id={patient_id} not found.")

            patient.diagnostic = new_diagnosys  # type: ignore
            session.commit()
            logger.info(f"Updated diagnosys for patient id={patient_id}.")
            patient = PatientSchema.model_validate(patient)

        return patient
