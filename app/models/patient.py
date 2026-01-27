"""
Patient Model Module.

This module defines the SQLAlchemy ORM model for patient records in a medical
triage system.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class Patient(Base):
    """
    SQLAlchemy model representing a patient record in the triage system.

    This model stores patient identification, triage severity level,
    disease classification, expressed symptoms, and vital signs.

    Attributes:
        id (int): Primary key, auto-incremented unique identifier.
        nom (str): Patient's last name (max 100 characters). Required.
        prenom (str): Patient's first name (max 100 characters). Required.
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nom: Mapped[str] = mapped_column(String(100), nullable=False)
    prenom: Mapped[str] = mapped_column(String(100), nullable=False)
    gravite: Mapped[str] = mapped_column(
        String(10),
        CheckConstraint("gravite IN ('gris', 'vert', 'jaune', 'rouge')"),
        nullable=False,
    )
    type_maladie: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    symptomes_exprimes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Vital signs
    fc: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Heart rate (bpm)
    fr: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Respiratory rate (cycles/min)
    spo2: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # Oxygen saturation (%)
    ta_systolique: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Systolic blood pressure (mmHg)
    ta_diastolique: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Diastolic blood pressure (mmHg)
    temperature: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # Body temperature (°C)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    archived: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # If the record is archived or not
    contexte: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Context for LLM containing diagnosis information
    diagnostic: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # Diagnosis information

    def __repr__(self):
        """Return a string representation of the Patient instance."""
        return f"<Patient(id={self.id}, nom='{self.nom}', prenom='{self.prenom}', gravite='{self.gravite}')>"
