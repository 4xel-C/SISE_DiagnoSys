from app.rag import VectorStore, document_store, patient_store
from app.services import PatientService


class RagServices:
    """Class to orchestrate the rag pipeline. Take text as a starting point from user input, manage patient context, retrieve documents, andd generate responses."""

    def __init__(
        self,
        document_store: VectorStore = document_store,
        patient_store: VectorStore = patient_store,
        patient_service: PatientService = PatientService(),
    ):
        # save the vectorstore
        self.document_store = document_store
        self.patient_store = patient_store
        self.patient_service = patient_service

    # TODO: Implement security and pertinency check function
    def check_pertinency_security(self):
        """TO BE IMPLEMENTED: Check pertinency and security of user input."""
        ...

    # TODO: To be implemented
    def compute_user_input(self, user_input: str):
        """
        Compute user input to update context, retrive best docuemnts, and closest case from the patients table.
        """
        ...
