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

from typing import Dict, List, Optional

from flask import render_template
from pydantic import BaseModel, computed_field


class PatientProcheSchema(BaseModel):
    """Schema for a related patient with similarity score."""

    patient_id: int
    similarity_score: Optional[float] = None

    model_config = {"from_attributes": True}


class DocumentProcheSchema(BaseModel):
    """Schema for a related document with similarity score."""

    document_id: int
    similarity_score: Optional[float] = None

    model_config = {"from_attributes": True}


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

    # Vital signs
    fc: Optional[int] = None
    fr: Optional[int] = None
    spo2: Optional[float] = None
    ta_systolique: Optional[int] = None
    ta_diastolique: Optional[int] = None
    temperature: Optional[float] = None
    contexte: Optional[str] = None
    diagnostic: Optional[str] = None

    # Related patients and documents (with similarity scores)
    patients_proches: List[PatientProcheSchema] = []
    documents_proches: List[DocumentProcheSchema] = []

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_with_relations(cls, patient_orm) -> "PatientSchema":
        """
        Create a PatientSchema from an ORM object, including relations with scores.

        Use this method instead of model_validate when relations are needed.

        Args:
            patient_orm: SQLAlchemy Patient object with loaded relationships.

        Returns:
            PatientSchema: Schema instance with relations populated.
        """
        return cls(
            id=patient_orm.id,
            nom=patient_orm.nom,
            prenom=patient_orm.prenom,
            gravite=patient_orm.gravite,
            type_maladie=patient_orm.type_maladie,
            symptomes_exprimes=patient_orm.symptomes_exprimes,
            fc=patient_orm.fc,
            fr=patient_orm.fr,
            spo2=patient_orm.spo2,
            ta_systolique=patient_orm.ta_systolique,
            ta_diastolique=patient_orm.ta_diastolique,
            temperature=patient_orm.temperature,
            contexte=patient_orm.contexte,
            diagnostic=patient_orm.diagnostic,
            patients_proches=[
                PatientProcheSchema(
                    patient_id=assoc.patient_proche_id,
                    similarity_score=assoc.similarity_score,
                )
                for assoc in patient_orm.patients_proches_assoc
            ],
            documents_proches=[
                DocumentProcheSchema(
                    document_id=assoc.document_id,
                    similarity_score=assoc.similarity_score,
                )
                for assoc in patient_orm.documents_proches_assoc
            ],
        )

    @computed_field
    @property
    def initials(self) -> str:
        """
        Generate initials from nom and prenom.

        Returns:
            str: Initials.
        """
        return f"{self.prenom[0].upper()}{self.nom[0].lower()}"

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

    def render(self, style="profile"):
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
                    name=self.prenom,
                    last_name=self.nom,
                    initials=self.initials,
                )
            case "case":
                return render_template("elements/patient_case.html")
            case _:
                raise ValueError(
                    f'Unexpected style argument "{style}". Should be "profile" or "case"'
                )
