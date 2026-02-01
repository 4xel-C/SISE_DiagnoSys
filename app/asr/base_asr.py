"""ASR service base class."""

from abc import ABC, abstractmethod


class ASRServiceBase(ABC):
    """
    Abstract base class for ASR services.

    Defines the interface that all ASR implementations must follow.
    """

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe complete audio data.

        Args:
            audio_data: Complete audio bytes to transcribe.

        Returns:
            str: The transcription text.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the ASR service is available and properly configured.

        Returns:
            bool: True if service is ready, False otherwise.
        """
