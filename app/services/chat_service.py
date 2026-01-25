"""
Chat Service Module.

This module provides a conversational service that simulates a patient
for training or playground purposes. It manages conversation history
and interacts with the LLM.

Example:
    >>> from app.services import ChatService
    >>> from app.schemas import PatientSchema
    >>> chat = ChatService(patient_schema)
    >>> response = chat.send_message("Bonjour, qu'est-ce qui vous amène ?")
    >>> print(response)
"""

import logging
from typing import Optional

from app.rag.llm import Message, llm_handler
from app.rag.llm_options import SYSTEM_PROMPT, SystemPromptTemplate
from app.schemas import PatientSchema
from app.services import PatientService

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service for managing patient simulation conversations.

    This service maintains conversation history and handles message
    exchange with the LLM, simulating a patient based on provided
    patient data.

    Attributes:
        patient: The patient schema with simulation data.
        system_prompt: Formatted system prompt with patient info.
        history: List of conversation messages.

    Example:
        >>> chat = ChatService(id)
        >>> response = chat.send_message("Comment vous sentez-vous ?")
        >>> print(chat.get_history())
    """

    def __init__(self, patient_id: int):
        """
        Initialize the chat service with a patient id.

        Args:
            patient_id: The ID of the patient to simulate.
        """
        patient_service = PatientService()
        patient: Optional[PatientSchema] = patient_service.get_by_id(patient_id)

        self.patient = patient

        # Prepare the system prompt with patient details
        self.system_prompt = SYSTEM_PROMPT[SystemPromptTemplate.CONVERSATION].format(
            nom=patient.nom,
            prenom=patient.prenom,
            symptomes=patient.symptomes_exprimes or "Non spécifiés",
        )

        # Initialize empty conversation history
        self.history: list[Message] = []

        logger.info(
            f"ChatService initialized for patient: {patient.prenom} {patient.nom}"
        )

        # Generate initial greeting from the patient
        self._send_initial_greeting()

    def _send_initial_greeting(self) -> None:
        """Generate the patient's initial greeting message."""
        greeting_prompt = "Présente-toi brièvement au médecin et explique pourquoi tu viens consulter."

        response = llm_handler.chat(
            new_message=greeting_prompt,
            history=[],
            system_prompt=self.system_prompt,
        )

        # Add only the assistant response to history (no user message for greeting)
        self.history.append({"role": "assistant", "content": response.content})

        logger.debug(f"Initial greeting generated: {response.content[:50]}...")

    def send_message(self, message: str) -> str:
        """
        Send a message and get the patient's response.

        Args:
            message: The doctor's message to the patient.

        Returns:
            The patient's response.
        """
        logger.debug(f"Sending message: {message[:50]}...")

        response = llm_handler.chat(
            new_message=message,
            history=self.history,
            system_prompt=self.system_prompt,
        )

        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response.content})

        logger.debug(f"Response received: {response.content[:50]}...")

        return response.content

    def get_history(self) -> list[Message]:
        """
        Get the conversation history.

        Returns:
            List of messages with 'role' and 'content' keys.
        """
        return self.history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared")

    def get_message_count(self) -> int:
        """
        Get the number of messages in history.

        Returns:
            Number of messages (user + assistant).
        """
        return len(self.history)
