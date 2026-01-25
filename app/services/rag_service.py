import logging
from typing import Any, Dict, List, Optional

from app.asr import ASRServiceBase, ASRServiceFactory

from app.rag import (
    LLMHandler,
    PromptTemplate,
    SystemPromptTemplate,
    VectorStore,
    document_store,
    guardrail_classifier,
    llm_handler,
    patient_store,
)
from app.schemas import PatientSchema
from app.services import DocumentService, PatientService

logger = logging.getLogger(__name__)


class UnsafeRequestException(Exception):
    """Custom exception for unsafe requests detected by guardrail."""

    def __init__(
        self,
        message: str,
        confidence: float = 0.0,
        checkpoint: str = "",
    ):
        super().__init__(message)
        self.confidence = confidence
        self.checkpoint = checkpoint

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "error": str(self),
            "code": "INJECTION_DETECTED",
            "confidence": self.confidence,
            "checkpoint": self.checkpoint,
        }


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
        asr_model: Optional[ASRServiceBase] = None,
    ):
        # save the instances of the core components
        self.document_vector_store = document_vector_store
        self.patient_vector_store = patient_vector_store
        self.patient_service = patient_service
        self.document_service = document_service
        self.llm_handler = llm_handler
        self.asr_model = asr_model if asr_model else ASRServiceFactory.create()

    def transcribe_stream(self, audio_chunk: bytes) -> Dict[str, Any]:
        """
        Transcribe a chunk of audio stream.

        Args:
            audio_chunk (bytes): The audio chunk to transcribe.

        Returns:
            Dict[str, Any]: The transcription result (partial or final).
        """
        return self.asr_model.transcribe_stream(audio_chunk)

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

        # Checkpoint 1: Check raw transcript before LLM summarization
        if not self.is_pertinent_and_secure(audio_input, checkpoint="raw_transcript"):
            logger.warning(
                "Raw transcript failed guardrail check, stopping context update: "
                + audio_input[: min(50, len(audio_input))]
                + "..."
            )
            raise UnsafeRequestException(
                "Raw transcript failed security check",
                checkpoint="raw_transcript",
            )

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

        # Checkpoint 2: Check synthesized context before storing/embedding
        if not self.is_pertinent_and_secure(
            new_context, checkpoint="synthesized_context"
        ):
            logger.warning(
                "Synthesized context failed guardrail check, stopping context update"
            )
            raise UnsafeRequestException(
                "Synthesized context failed security check",
                checkpoint="synthesized_context",
            )

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
                    "related_documents": List[int, float],
                    "related_patients": List[int, float],
                }
        """
        # retrieve patient context from DB for embeddings
        patient: PatientSchema = self.patient_service.get_by_id(patient_id=patient_id)

        if not patient:
            raise ValueError(f"Patient with id={patient_id} not found.")

        logger.debug(
            f"compute_rag_diagnosys: Retrieved patient context for patient id={patient_id}."
        )

        # ================================================================ Document retrieval
        # find top 5 relevant document chunks from vector store
        document_chunks = self.find_topk_similar_documents(
            patient.content_for_embedding, k=5
        )

        chunk_text = [chunk["text"] for chunk in document_chunks]

        # ================================================================ Diagnosys generation
        # Checkpoint 3: Check patient context before diagnosis LLM call
        if patient.contexte and not self.is_pertinent_and_secure(
            patient.contexte, checkpoint="pre_diagnosis"
        ):
            logger.warning(
                f"Patient context failed guardrail check before diagnosis, "
                f"patient_id={patient_id}"
            )
            raise UnsafeRequestException(
                "Patient context failed security check before diagnosis generation",
                checkpoint="pre_diagnosis",
            )

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

        return {"diagnosys": diagnosys_text}

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

    # TODO : Add pertinency check with a classifier to avoid useless calls to the llm
    def is_pertinent_and_secure(
        self, user_input: str, checkpoint: str = "raw_input"
    ) -> bool:
        """
        Check if user input is safe from prompt injection attacks.

        Args:
            user_input: The text to validate.
            checkpoint: Identifier for where the check is performed (for logging).

        Returns:
            True if the input is safe, False if injection detected.
        """
        if not user_input or not user_input.strip():
            return True

        result = guardrail_classifier.predict(user_input)

        if result.is_injection:
            logger.warning(
                f"Guardrail [{checkpoint}]: Injection detected "
                f"(confidence={result.confidence:.3f}), "
                f"input_preview='{user_input[:50]}...'"
            )
            return False

        logger.debug(
            f"Guardrail [{checkpoint}]: Input cleared (confidence={result.confidence:.3f})"
        )
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
