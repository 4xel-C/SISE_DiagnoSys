from app.schemas.document_schema import DocumentSchema
from app.schemas.patient_schema import (
    DocumentProcheSchema,
    PatientProcheSchema,
    PatientSchema,
)
from app.schemas.scraped_document import ScrapedDocument

__all__ = [
    "PatientSchema",
    "ScrapedDocument",
    "DocumentSchema",
    "PatientProcheSchema",
    "DocumentProcheSchema",
]
