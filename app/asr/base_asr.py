"""ASR service base class."""

from abc import ABC, abstractmethod
import io
import wave

import numpy as np


class ASRServiceBase(ABC):
    """
    Abstract base class for ASR services.

    Defines the interface that all ASR implementations must follow.
    """

    @abstractmethod
    def transcribe_stream(self, audio_chunk: bytes) -> dict[str, str | bool]:
        """
        Process a chunk of audio for streaming transcription.

        Args:
            audio_chunk: A chunk of audio bytes.

        Returns:
            dict: Partial or final transcription result.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the ASR service is available and properly configured.

        Returns:
            bool: True if service is ready, False otherwise.
        """
