"""
Patient Model Module.

This module defines the SQLAlchemy ORM model for patient records in a medical
triage system.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from app.models import Base


class Patient(Base):
    """
    SQLAlchemy model representing a patient record in the triage system.

    This model stores patient identification, triage severity level,
    disease classification, expressed symptoms, and vital signs.

    Attributes:
        id (int): Primary key, auto-incremented unique identifier.
        nom (str): Patient's last name (max 100 characters). Required.
        prenom (str): Patient's first name (max 100 characters). Optional.
        gravite (str): Triage severity level. Must be one of:
            - 'gris' (gray): Deceased or palliative care
            - 'vert' (green): Minor injuries, can wait
            - 'jaune' (yellow): Serious but stable, delayed priority
            - 'rouge' (red): Immediate life-threatening emergency
        type_maladie (str): Disease type/category (max 200 characters). Optional.
        symptomes_exprimes (str): Free-text description of expressed symptoms. Optional.
        fc (int): Heart rate in beats per minute (bpm). Optional.
        fr (int): Respiratory rate in cycles per minute. Optional.
        spo2 (float): Oxygen saturation percentage (0-100%). Optional.
        ta_systolique (int): Systolic blood pressure in mmHg. Optional.
        ta_diastolique (int): Diastolic blood pressure in mmHg. Optional.
        temperature (float): Body temperature in Celsius. Optional.
        created_at (datetime): Timestamp of record creation. Auto-generated.
    """

    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(100), nullable=False)
    prenom = Column(String(100), nullable=True)
    gravite = Column(
        String(10),
        CheckConstraint("gravite IN ('gris', 'vert', 'jaune', 'rouge')"),
        nullable=False,
    )
    type_maladie = Column(String(200), nullable=True)
    symptomes_exprimes = Column(Text, nullable=True)

    # Vital signs
    fc = Column(Integer, nullable=True)  # Heart rate (bpm)
    fr = Column(Integer, nullable=True)  # Respiratory rate (cycles/min)
    spo2 = Column(Float, nullable=True)  # Oxygen saturation (%)
    ta_systolique = Column(Integer, nullable=True)  # Systolic blood pressure (mmHg)
    ta_diastolique = Column(Integer, nullable=True)  # Diastolic blood pressure (mmHg)
    temperature = Column(Float, nullable=True)  # Body temperature (°C)

    created_at = Column(DateTime, default=datetime.now)
    archived = Column(Boolean, default=False)  # If the record is archived or not
    contexte = Column(
        Text, nullable=True
    )  # Context for LLM containing diagnosis information

    def __repr__(self):
        """Return a string representation of the Patient instance."""
        return f"<Patient(id={self.id}, nom='{self.nom}', gravite='{self.gravite}')>"
