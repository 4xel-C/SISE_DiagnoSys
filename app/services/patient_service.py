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

from app.config import Database, db
from app.models import Patient
from app.schemas import PatientSchema

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
    # CREATE METHODS
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
        with self.db_manager.session() as session:
            patient = Patient(**kwargs)
            session.add(patient)
            session.commit()
            logger.info(f"Created patient: {patient}")
            return PatientSchema.model_validate(patient)

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
                return True
            logger.debug(f"Patient with id={patient_id} not found for deletion.")
            return False
