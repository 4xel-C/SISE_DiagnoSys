import logging
from typing import Any, Dict, List, Optional

from app.asr.base import ASRServiceBase
from app.asr.factory import ASRServiceFactory
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
        asr_service: Optional[ASRServiceBase] = None,
    ):
        # save the instances of the core components
        self.document_vector_store = document_vector_store
        self.patient_vector_store = patient_vector_store
        self.patient_service = patient_service
        self.document_service = document_service
        self.llm_handler = llm_handler
        self.asr_service = asr_service if asr_service else ASRServiceFactory.create()

    def transcribe_stream(self, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Transcribe a chunk of audio stream.

        Args:
            audio_chunk (bytes): The audio chunk to transcribe.

        Returns:
            Dict[str, Any]: The transcription result (partial or final).
        """
        return self.asr_service.transcribe_stream(audio_chunk)

    def update_context_after_audio(
        self,
        id: int,
        audio_input: str,
    ) -> str:
        """From the user input (context modification or speech to text addition to context), return a condensed context to update the patient context.
        The new context if then update in the database before beeing return as a string for further use in the application.
        Args:
            user_input (str): The input text from the user.
            from_audio (bool): Whether the input comes from audio transcription.
            id_patient (str): The ID of the patient whose context is to be updated.
        Returns:
            str: The updated context string
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

        return updated_patient.contexte  # type: ignore

    def compute_rag_diagnosys(self, patient_id: int) -> Dict[str, Any]:
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

        if not patient:
            raise ValueError(f"Patient with id={patient_id} not found.")

        logger.debug(
            f"compute_rag_diagnosys: Retrieved patient context for patient id={patient_id}."
        )

        # ================================================================ Patient retrieval
        close_patients = self.find_topk_similar_patients(
            patient_id=patient_id, context=patient.content_for_embedding, k=5
        )
        patient_ids = [int(p["metadata"]["patient_id"]) for p in close_patients]

        patient_scores = [float(p["similarity"]) for p in close_patients]

        # update database
        self.patient_service.update_close_patients(
            patient_id=patient_id,
            close_patient_ids=patient_ids,
            similarity_scores=patient_scores,
        )

        # ================================================================ Document retrieval
        # find top 5 relevant document chunks from vector store
        document_chunks = self.find_topk_similar_documents(
            patient.content_for_embedding, k=5
        )

        chunk_text = [chunk["text"] for chunk in document_chunks]
        document_ids = list(
            {int(chunk["metadata"]["document_id"]) for chunk in document_chunks}
        )

        document_scores = [float(chunk["similarity"]) for chunk in document_chunks]

        self.patient_service.update_documents(
            patient_id=patient_id,
            document_ids=document_ids,
            similarity_scores=document_scores,
        )

        # ================================================================ Diagnosys generation
        diagnosys_text = self.generate_diagnosys(
            context=patient.contexte
            if patient.contexte
            else "Pas de contexte disponible.",
            documents_chunks=chunk_text,
        )

        if not diagnosys_text:
            raise LLMGenerationException("Failed to generate diagnosys.")

        # add the diagnosys to the database
        self.patient_service.update_diagnosys(
            patient_id=patient_id, new_diagnosys=diagnosys_text
        )

        return {
            "diagnosys": diagnosys_text,
            "document_ids": document_ids,
            "document_scores": document_scores,
            "related_patients_ids": patient_ids,
            "related_patients_scores": patient_scores,
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

    def find_topk_similar_patients(
        self, patient_id: int, context: str, k: int = 5
    ) -> List[Dict]:
        """
        Find top-k similar patients based on patient context.

        Args:
            patient_id (int): The patient's unique identifier.
            context (str): The patient's medical context.
            k (int): Number of similar patients to retrieve.
        Returns:
            List[Dict]: List of similar patient dictionaries with metadata.
        """
        patient_results = self.patient_vector_store.search(
            context, n_results=k, where={"patient_id": {"$ne": patient_id}}
        )
        return patient_results

    def find_topk_similar_documents(self, context: str, k: int = 5) -> List[Dict]:
        """
        Find top-k similar documents based on input context.

        Args:
            context (str): The input text context.
            k (int): Number of similar documents to retrieve.
        Returns:
            List[Dict]: List of similar document chunks with metadata (vector_store provide the key from the search method).
        """
        document_chunks: List[Dict] = self.document_vector_store.search(
            context,
            n_results=k,
        )

        # seens document id:
        seen_document_ids = set()

        unique_document_chunks = list()

        # delete duplicates from the results while preserving order
        for chunk in document_chunks:
            doc_id = int(chunk["metadata"]["document_id"])
            if doc_id in seen_document_ids:
                continue
            else:
                seen_document_ids.add(doc_id)

        return unique_document_chunks

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
