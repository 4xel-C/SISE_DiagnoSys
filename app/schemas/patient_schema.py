"""
Patient Schema Module for Vector Database.

This module defines Pydantic schemas for transforming Patient ORM models
into data suitable for ChromaDB vector storage.

Example:
    >>> from app.models.patient import Patient
    >>> from app.schemas.patient_schema import PatientSchema
    >>> patient_orm = session.query(Patient).get(1)
    >>> schema = PatientSchema.model_validate(patient_orm)
    >>> schema.vector_id
    'patient_1'
"""

from datetime import datetime
from typing import Optional

from flask import render_template
from pydantic import BaseModel, computed_field


class PatientSchema(BaseModel):
    """
    Pydantic schema for patient data indexing in ChromaDB.

    This schema transforms SQLAlchemy Patient model into a format suitable
    for vector database storage, with methods for generating embedding content
    and metadata.

    Attributes:
        id (int): Patient primary key from database.
        nom (str): Patient's last name.
        prenom (str, optional): Patient's first name.
        gravite (str, optional): Triage severity level.
        type_maladie (str, optional): Disease type/category.
        symptomes_exprimes (str, optional): Expressed symptoms description.
        fc (int, optional): Heart rate (bpm).
        fr (int, optional): Respiratory rate.
        spo2 (float, optional): Oxygen saturation (%).
        ta_systolique (int, optional): Systolic blood pressure (mmHg).
        ta_diastolique (int, optional): Diastolic blood pressure (mmHg).
        temperature (float, optional): Body temperature (°C).

    Example:
        >>> schema = PatientSchema.model_validate(patient_orm)
        >>> collection.add(
        ...     ids=[schema.vector_id],
        ...     documents=[schema.content_for_embedding],
        ...     metadatas=[schema.to_metadata()]
        ... )
    """

    id: int
    nom: str
    prenom: str
    gravite: Optional[str] = None
    type_maladie: Optional[str] = None
    symptomes_exprimes: Optional[str] = None
    created_at: Optional[datetime] = None

    # Vital signs
    fc: Optional[int] = None
    fr: Optional[int] = None
    spo2: Optional[float] = None
    ta_systolique: Optional[int] = None
    ta_diastolique: Optional[int] = None
    temperature: Optional[float] = None
    contexte: Optional[str] = None
    diagnostic: Optional[str] = None

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def initials(self) -> str:
        """
        Generate initials from nom and prenom.

        Returns:
            str: Initials.
        """
        return f"{self.nom[0].upper()}{self.prenom[0].lower()}"
    
    @computed_field
    @property
    def formatted_date(self) -> str:
        """
        Format created_at date to dd-mm-yy

        Returns
         str: formated date
        """
        if not self.created_at:
            return ""
        return self.created_at.strftime('%d-%m-%y')

    @computed_field
    @property
    def vector_id(self) -> str:
        """
        Generate unique identifier for ChromaDB.

        Returns:
            str: Unique ID in format 'patient_{id}'.
        """
        return f"patient_{self.id}"

    @computed_field
    @property
    def content_for_embedding(self) -> str:
        """
        Generate text content to be vectorized.

        Combines relevant patient information into a single string
        suitable for embedding generation.

        Returns:
            str: Concatenated patient information for vectorization.
        """
        parts = []

        if self.symptomes_exprimes:
            parts.append(f"Symptômes: {self.symptomes_exprimes}")

        if self.type_maladie:
            parts.append(f"Maladie: {self.type_maladie}")

        if self.contexte:
            parts.append(f"Contexte: {self.contexte}")

        if self.diagnostic:
            parts.append(f"Diagnostic: {self.diagnostic}")

        return ". ".join(parts) if parts else ""

    def to_metadata(self) -> dict:
        """
        Generate metadata dictionary for ChromaDB storage.

        Returns:
            dict: Metadata with patient identifiers and triage info.
                  Empty strings used for None values (ChromaDB requirement).
        """
        return {
            "patient_id": self.id,
            "nom": self.nom,
            "prenom": self.prenom or "",
            "gravite": self.gravite or "",
            "type_maladie": self.type_maladie or "",
            "contexte": self.contexte or "",
        }

    def render(self, style="profile", **kwargs):
        """
        Render a HTML template from the patient

        Arguments:
            style (str): The template style to render. Value should be in ["profile", "case"]

        Returns:
            str: HTML string
        """
        match style:
            case "profile":
                return render_template(
                    "elements/patient_profile.html",
                    id=self.id,
                    first_name=self.prenom,
                    last_name=self.nom,
                    initials=self.initials,
                    **kwargs,
                )
            case "case":
                return render_template(
                    "elements/patient_case.html",
                    id=self.id,
                    first_name=self.prenom,
                    last_name=self.nom,
                    initials=self.initials,
                    date=self.formatted_date,
                    **kwargs,
                )
            case _:
                raise ValueError(
                    f'Unexpected style argument "{style}". Should be "profile" or "case"'
                )
