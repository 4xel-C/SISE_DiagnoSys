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
from typing import List, Optional, Tuple

from sqlalchemy import and_, or_

from app.config import Database, db
from app.models import Patient
from app.rag import document_store, patient_store
from app.schemas import Gravite, PatientSchema

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
            return patient.contexte if patient.contexte else ""

    def get_diagnostic(self, patient_id: int) -> str:
        """
        Retrieve the diagnostic field of a patient.

        Args:
            patient_id (int): The patient's unique identifier.

        Returns:
            str: The diagnostic of the patient.

        Example:
            >>> diagnostic = service.get_diagnostic(1)
        """
        logger.debug(f"Fetching diagnostic for patient id={patient_id}.")
        with self.db_manager.session() as session:
            patient = session.query(Patient).filter_by(id=patient_id).first()
            if not patient:
                logger.error(
                    f"Patient with id={patient_id} not found for diagnostic retrieval."
                )
                raise ValueError(f"Patient with id={patient_id} not found.")

            logger.debug(f"Retrieved diagnostic for patient id={patient_id}.")
            return patient.diagnostic if patient.diagnostic else ""

    def get_documents_proches(
        self, patient_id: int, n=5
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Retrieve related documents for a patient, sorted by similarity score.
        The proximity from the documents are directly calculated from the Chromadb.

        Args:
            patient_id (int): The patient's unique identifier.
            n (int): Number of document chunks to retrieve (default = 5).

        Returns:
            Optional[List[Tuple[int, float]]]: A list of the closest document IDs with their similarity scores.

        Raises:
            ValueError: If the patient is not found.

        Example:
            >>> docs = service.get_documents_proches(1)
        """
        logger.debug(f"Fetching related documents for patient id={patient_id}.")

        # get the embedding of the patient
        embedding: Optional[List[float]] = (
            patient_store.get(patient_id)[0].get("embedding")
            if patient_store.get(patient_id)
            else None
        )

        if embedding is None:
            return None

        document_chunks = document_store.search(
            embedding,
            n_results=n,
            embed=True,
        )

        # seens document id:
        document_ids = list()
        similarity_scores = list()

        # delete duplicates from the results while preserving order
        for chunk in document_chunks:
            doc_id = int(chunk["metadata"]["document_id"])
            if doc_id in document_ids:
                continue
            else:
                document_ids.append(doc_id)
                similarity_scores.append(chunk["similarity"])

        return list(zip(document_ids, similarity_scores))

    def get_patients_proches(
        self, patient_id: int, n=5
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Retrieve a set of patient having a similar medical context with their similarity score.
        The proximity from the documents are directly calculated from the Chromadb.

        Args:
            patient_id (int): The patient's unique identifier.
            n (int): The number of patients to retrieve.

        Returns:
            List[Tuple[int, float]]: A list containing the patients ids with their similarity score.

        Raises:
            ValueError: If the patient is not found.

        Example:
            >>> patients = service.get_patients_proches(1)
        """
        logger.debug(f"Fetching related patients context for patient id={patient_id}.")

        # get the embedding of the patient
        embedding: Optional[List[float]] = (
            patient_store.get(patient_id)[0].get("embedding")
            if patient_store.get(patient_id)
            else None
        )

        if embedding is None:
            return None

        patient_results = patient_store.search(
            embedding,
            n_results=n,
            where={"patient_id": {"$ne": patient_id}},
            embed=True,
        )

        patient_ids = [
            patient["metadata"].get("patient_id", []) for patient in patient_results
        ]
        similarity_scores = [patient["similarity"] for patient in patient_results]

        return list(zip(patient_ids, similarity_scores))

    ################################################################
    # CREATE METHODS
    ################################################################

    def create(
        self,
        nom: str,
        prenom: str,
        gravite: Optional[str] = None,
        type_maladie: Optional[str] = None,
        symptomes_exprimes: Optional[str] = None,
        fc: Optional[int] = None,
        fr: Optional[int] = None,
        spo2: Optional[float] = None,
        ta_systolique: Optional[int] = None,
        ta_diastolique: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> PatientSchema:
        """
        Create a new patient record.
        With respect to the Patient model fields.

        Args:
            nom (str): Patient's last name.
            prenom (str): Patient's first name.
            gravite (Optional[Gravite], optional): Triage severity level. Defaults to None.
            type_maladie (Optional[str], optional): Disease type/category. Defaults to None.
            symptomes_exprimes (Optional[str], optional): Expressed symptoms description. Defaults to None.
            fc (Optional[int], optional): Heart rate (bpm). Defaults to None.
            fr (Optional[int], optional): Respiratory rate. Defaults to None.
            spo2 (Optional[float], optional): Oxygen saturation (%). Defaults to None.
            ta_systolique (Optional[int], optional): Systolic blood pressure (mmHg). Defaults to None.
            ta_diastolique (Optional[int], optional): Diastolic blood pressure (mmHg). Defaults to None.
            temperature (Optional[float], optional): Body temperature (°C). Defaults to None.

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
        logger.debug(f"Creating new patient with data: {nom}, {prenom}.")
        patient_schema: PatientSchema

        gravite_enum = Gravite(gravite) if gravite else None

        with self.db_manager.session() as session:
            patient = Patient(
                nom=nom,
                prenom=prenom,
                gravite=gravite_enum.value if gravite_enum else None,
                type_maladie=type_maladie,
                symptomes_exprimes=symptomes_exprimes,
                fc=fc,
                fr=fr,
                spo2=spo2,
                ta_systolique=ta_systolique,
                ta_diastolique=ta_diastolique,
                temperature=temperature,
            )

            session.add(patient)
            session.commit()
            session.flush()
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

            patient.contexte = new_context
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

            patient.diagnostic = new_diagnosys
            session.commit()
            logger.info(f"Updated diagnosys for patient id={patient_id}.")
            patient = PatientSchema.model_validate(patient)

        return patient
