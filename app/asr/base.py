"""ASR service base class."""

from abc import ABC, abstractmethod

import numpy as np


class ASRServiceBase(ABC):
    """
    Abstract base class for ASR services.

    Defines the interface that all ASR implementations must follow.
    """

    def transform_chunk_audio_to(self, which: str, chunk_float: bytes) -> bytes:
        """
        Transform a chunk of audio to the specified format.

        Args:
            which: Target audio format (e.g., 'int16', 'float32').
            chunk_float: Audio chunk in float format.

        Returns:
            bytes: Transformed audio chunk.
        """
        match which:
            case "int16":
                audio_int16 = (
                    np.frombuffer(chunk_float, dtype=np.float32) * 32768
                ).astype(np.int16)
                return audio_int16.tobytes()

            case "float32":
                return chunk_float

            case _:
                raise ValueError(f"Unsupported audio format: {which}")

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
