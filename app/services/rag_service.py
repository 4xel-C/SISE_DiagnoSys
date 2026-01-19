import logging
from typing import Any, Dict, List, Optional, Tuple

from app.rag import (
    LLMHandler,
    PromptTemplate,
    SystemPromptTemplate,
    VectorStore,
    document_store,
    llm_handler,
    patient_store,
)
from app.schemas import PatientSchema
from app.services import DocumentService, PatientService

logger = logging.getLogger(__name__)


class UnsafeRequestException(Exception):
    """Custom exception for unsafe requests in RAG service."""

    pass


class LLMGenerationException(Exception):
    """Custom exception for LLM generation failures in RAG service."""

    pass


class RagService:
    """Class to orchestrate the rag pipeline. Take text as a starting point from user input, manage patient context, retrieve documents, andd generate responses."""

    def __init__(
        self,
        document_vector_store: VectorStore = document_store,
        patient_vector_store: VectorStore = patient_store,
        patient_service: PatientService = PatientService(),
        document_service: DocumentService = DocumentService(),
        llm_handler: LLMHandler = llm_handler,
    ):
        # save the instances of the core components
        self.document_vector_store = document_vector_store
        self.patient_vector_store = patient_vector_store
        self.patient_service = patient_service
        self.document_service = document_service
        self.llm_handler = llm_handler

    def update_context_after_audio(
        self,
        id: int,
        audio_input: str,
    ) -> Optional[str]:
        """From the user input (context modification or speech to text addition to context), return a condensed context to update the patient context.
        The new context if then update in the database before beeing return as a string for further use in the application.
        Args:
            user_input (str): The input text from the user.
            from_audio (bool): Whether the input comes from audio transcription.
            id_patient (str): The ID of the patient whose context is to be updated.
        Returns:
            Optional[str]: The updated context string, or None if the input was not pertinent or secure
        """

        # check pertinency and guardrail -> if user input not pertinent, stop the pipeline, as the context would not be refreshed.
        if not self.is_pertinent_and_secure(audio_input):
            logger.warning(
                "Audio input not pertinent or secure, stopping context update: "
                + audio_input[: min(50, len(audio_input))]
                + "..."
            )
            raise UnsafeRequestException("Audio input not pertinent or secure.")

        # retrieve the patient context from db
        patient = self.patient_service.get_by_id(patient_id=id)

        # generate a condensed context from patient context and user input
        new_context = self.llm_handler.generate_with_template(
            template=PromptTemplate.SUMMARY,
            system_prompt=SystemPromptTemplate.CONTEXT_UPDATER,
            context=patient.contexte,
            audio=audio_input,
        )

        if new_context.content:
            logger.info("Context successfully updated with audio input")
            new_context = new_context.content.strip()
        else:
            logger.error("llm_handler failed to generate updated context")
            raise LLMGenerationException("Failed to generate updated context")

        # update the patient context in the database
        updated_patient = self.patient_service.update_context(
            patient_id=id, new_context=new_context
        )

        return updated_patient.contexte

    def compute_rag_diagnosys(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """Generate diagnosys based on patient context and his metadata:
        - retrieve embedding from chromadb vector store
        - Retrieve top5 relevant document chunks from vector store and return document ids
        - Retrieve top5 relevant patient context and return closets patients ids.
        - Generate diagnosys

        Returns:
            Dict[str, str]: diagnosys response from LLM with the following structure:
                {
                    "diagnosys": str,
                    "document_ids": List[int],
                    "related_patients_ids": List[int],
                }
        """
        # retrieve patient context from DB for embeddings
        patient: PatientSchema = self.patient_service.get_by_id(patient_id=patient_id)
        patient_chroma = self.patient_vector_store.get(patient.vector_id)

        logger.debug(
            f"compute_rag_diagnosys: Retrieved patient context for patient id={patient_id}."
        )

        if not patient_chroma:
            raise ValueError(f"No chroma patient with id={patient_id} found.")

        patient_vector = patient_chroma[0].get("embedding", [])

        if not patient_vector:
            raise ValueError(f"No embedding found for patient id={patient_id}.")

        # ================================================================ Document retrieval
        # find top 5 relevant document chunks from vector store
        document_ids, document_chunks = self.find_topk_similar_documents_ids(
            patient_vector, k=5
        )

        # ================================================================ Patient retrieval
        patient_ids = self.find_topk_similar_patients_ids(patient_vector, k=5)

        diagnosys_text = self.generate_diagnosys(
            context=patient.contexte
            if patient.contexte
            else "Pas de contexte disponible.",
            documents_chunks=[res["document"] for res in document_chunks],
        )

        return {
            "diagnosys": diagnosys_text,
            "document_ids": document_ids,
            "related_patients_ids": patient_ids,
        }

    def generate_diagnosys(self, context: str, documents_chunks: List[str]) -> str:
        """
        Generate diagnosys based on patient context and relevant document chunks.

        Args:
            context (str): The patient's medical context.
            documents_chunks (List[str]): List of relevant document chunks.
        Returns:
            str: The generated diagnosys response from the LLM.
        """

        # generate diagnosys from LLM
        diagnosys_response = self.llm_handler.generate_with_template(
            template=PromptTemplate.DIAGNOSTIC,
            system_prompt=SystemPromptTemplate.DIAGNOSYS_ASSISTANT,
            context=context,
            documents_chunks=documents_chunks,
        )

        if diagnosys_response.content:
            diagnosys_text = diagnosys_response.content.strip()
            logger.info("Diagnosys successfully generated from LLM.")
        else:
            logger.error("llm_handler failed to generate diagnosys.")
            diagnosys_text = ""

        return diagnosys_text

    def find_topk_similar_patients_ids(self, context: str, k: int = 5) -> List[int]:
        """
        Find top-k similar patients based on patient context.

        Args:
            context (str): The patient's medical context.
            k (int): Number of similar patients to retrieve.
        Returns:
            List[int]: List of similar patient IDs.
        """
        patient_results = self.patient_vector_store.search(
            context,
            n_results=k,
        )
        patient_ids = list(
            {int(result["metadata"]["patient_id"]) for result in patient_results}
        )
        return patient_ids

    def find_topk_similar_documents_ids(
        self, context: str, k: int = 5
    ) -> Tuple[List[int], List[dict]]:
        """
        Find top-k similar documents based on input context.

        Args:
            context (str): The input text context.
            k (int): Number of similar documents to retrieve.
        Returns:
            Tuple[List[int], List[dict]]: A tuple containing a list of similar document IDs and the corresponding document chunks.
        """
        document_chunks = self.document_vector_store.search(
            context,
            n_results=k,
        )
        document_ids = list(
            {int(result["metadata"]["document_id"]) for result in document_chunks}
        )

        return document_ids, document_chunks

    # TODO: Implement security and pertinency check function
    def is_pertinent_and_secure(self, user_input: str) -> bool:
        """TO BE IMPLEMENTED: Check pertinency and security of user input."""
        return True

    # TODO: Connect translator from Olivier, to be ignored if we use a multinlingual embedder ?
    def translate_text(self, text: str) -> str:
        """
        translate user text (french) into english to be consistent with embedder used.
        Args:
            user_input (str): The input text from the user.

        Returns:
            str: The generated response.
        """
        return text
