import logging

from app.rag import LLMHandler, VectorStore, document_store, llm_handler, patient_store
from app.schemas import PatientSchema
from app.services import DocumentService, PatientService

logger = logging.getLogger(__name__)


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

    # TODO: Finish compute user input pipeline
    def compute_user_input(self, user_input: str, id_patient: int):
        """Send the user input into the RAG pipeline

        Args:
            user_input (str):
            id_patient (int): _description_

        Returns:
            _type_: None
        """
        # translation in english to be consistent with embeddings
        input_english = self.translate_text(user_input)
        logger.debug(
            "Translated context: " + input_english[: min(len(input_english), 50)]
        )

        # check pertinency and guardrail -> if user input not pertinent, stop the pipeline, as the context would not be refreshed.
        if not self.is_pertinent_and_secure(input_english):
            logger.warning("User input failed pertinency and safety check.")
            return None

        # retriveve patient
        patient: PatientSchema = self.patient_service.get_by_id(id_patient)

        # retrieve patient context.
        patient_context: str = patient.contexte if patient.contexte else ""

        context_english = ""
        if patient_context:
            # context translation
            context_english = self.translate_text(patient_context)
            logger.debug(
                "Translated context: "
                + context_english[: min(len(context_english), 50)]
            )

        logger.debug(f"Patient recuperated: {patient.nom}, {patient.prenom}")

        # combine context and user_input
        full_context = context_english + " " + input_english
        full_context = full_context.strip()

        # search for closest patients:
        closest_patient = self.patient_vector_store.search(
            query=full_context, n_results=5
        )
        logger.debug(f"Closest patients found: {len(closest_patient)}")

        # search for closest documents:
        closest_documents = self.document_vector_store.search(
            query=full_context, n_results=5
        )
        logger.debug(f"Closest documents found: {len(closest_documents)}")

        # prepare the prompt for llm
        prompt_kwargs = {
            "context": context_english,
            "documents_chunks": " ".join([doc["text"] for doc in closest_documents]),
            "patient_chunks": " ".join([pat["text"] for pat in closest_patient]),
            "query": input_english,
        }

        # generate the response
        llm_response = self.llm_handler.generate_with_template(**prompt_kwargs)

        logger.debug(
            "LLM Response: "
            + llm_response.content[: min(len(llm_response.content), 50)]
        )

        ...

    # TODO: Implement security and pertinency check function
    def is_pertinent_and_secure(self, user_input: str) -> bool:
        """TO BE IMPLEMENTED: Check pertinency and security of user input."""
        return True

    # TODO: Connect translator from Olivier
    def translate_text(self, text: str) -> str:
        """
        translate user text (french) into english to be consistent with embedder used.
        Args:
            user_input (str): The input text from the user.

        Returns:
            str: The generated response.
        """
        return text
