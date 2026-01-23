"""
ASR Service Module. (ASR: Automatic Speech Recognition)

This module provides methods for handling ASR-related operations.
Supports two backends:
- Vosk: Offline speech recognition (when online_mode=1)
- Kyutai (Moshi): Online speech recognition (when online_mode=0)

Example:
    >>> #TODO: from app.services.asr_service import ASRServiceFactory
    >>> asr_service = ASRServiceFactory.create()
    >>> text = asr_service.transcribe(audio_data)
    >>> print(text)
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Protocol, cast

import numpy as np
from dotenv import load_dotenv
from vosk import KaldiRecognizer, Model

load_dotenv()

logger = logging.getLogger(__name__)


class VoskRecognizerProtocol(Protocol):
    """
    Protocol for Vosk KaldiRecognizer methods used in ASR service.
    The methods are defined here for type checking purposes,
    because the vosk package lacks proper type hints.
    """

    def AcceptWaveform(self, data: bytes) -> bool: ...
    def Result(self) -> str: ...
    def PartialResult(self) -> str: ...
    def FinalResult(self) -> str: ...
    def SetWords(self, enable_words: bool) -> None: ...
    def SetPartialWords(self, enable_partial_words: bool) -> None: ...


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


type ASRService = type[ASRServiceBase]
type ASRAnswer = dict[str, str | bool]


class ASRServiceFactory:
    """
    Factory/registry for ASR services.

    Register implementations with:
        @ASRServiceFactory.register("vosk")
        class VoskASRService(ASRServiceBase): ...
    """

    _registry: dict[str, ASRService] = {}

    # Mapping of ONLINE_MODE env var to service name
    _online_offline_map: dict[int, str] = {
        0: "kyutai",
        1: "vosk",
    }

    @classmethod
    def register(cls, name: str):
        """Decorator to register an ASR service class under a given name."""
        key = name.lower().strip()

        def decorator(service_cls: ASRService) -> ASRService:
            if key in cls._registry:
                raise ValueError(f"ASR service '{key}' is already registered.")
            cls._registry[key] = service_cls
            return service_cls

        return decorator

    @classmethod
    def create(cls, which: str | None = None) -> ASRServiceBase:
        """
        Create an ASR service instance based on the specified type or environment configuration.

        Args:
            which: "kyutai" / "vosk" / etc. If None, uses ONLINE_MODE mapping.

        Raises:
            ValueError: If ONLINE_MODE is invalid or service name is unknown.
            NotImplementedError: If the requested service is not registered.

        Returns:
            ASRServiceBase: An instance of the requested ASR service.
        """
        if which is None:
            online_mode = int(os.getenv("ONLINE_MODE", "1"))
            which = cls._online_offline_map.get(online_mode, None)
            if which is None:
                logger.error(
                    "Invalid ONLINE_MODE=%d; cannot determine ASR service.", online_mode
                )
                raise ValueError(
                    f"Unsupported ONLINE_MODE={online_mode}. Expected one of: {list(cls._online_offline_map)}"
                )

        key = which.lower().strip()
        service_cls = cls._registry.get(key)
        if service_cls is None:
            logger.error(
                "ASR service '%s' is not registered. Available services: %s",
                key,
                sorted(cls._registry.keys()),
            )
            raise NotImplementedError(
                f"ASR service '{key}' is not registered. Registered services: {sorted(cls._registry.keys())}"
            )
        return service_cls()

    @classmethod
    def available(cls, which: str | None = None) -> bool:
        """Check if the specified ASR service is available."""
        try:
            service = cls.create(which)
            return service.is_available()
        except (
            NotImplementedError,
            ValueError,
            OSError,
            RuntimeError,
            FileNotFoundError,
        ):
            logger.error("ASR service '%s' is not available.", which)
            return False


@ASRServiceFactory.register("vosk")
class VoskASRService(ASRServiceBase):
    """
    Vosk ASR Service implementation.
    Uses the Vosk offline speech recognition engine.
    Primarly used for the online version of the app due to its small size.
    #TODO: example et tout
    """

    VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "data/vosk-model-small-fr-0.22")
    SAMPLE_RATE = 16000

    def __init__(self):
        self._model = Model(self.VOSK_MODEL_PATH)
        self._recognizer = cast(
            VoskRecognizerProtocol, KaldiRecognizer(self._model, self.SAMPLE_RATE)
        )
        self._recognizer.SetWords(enable_words=True)
        self._recognizer.SetPartialWords(enable_partial_words=True)
        logger.debug(
            "Vosk ASR Service initialized with model at %s", self.VOSK_MODEL_PATH
        )

    def transcribe_stream(self, audio_chunk: bytes) -> ASRAnswer:
        if len(audio_chunk) == 0:
            return {"partial": "", "final": False}

        if self._recognizer.AcceptWaveform(audio_chunk):
            result = json.loads(self._recognizer.Result()).get("text", "")
            logger.debug("Vosk final result: %s", result)
            return {"text": result, "final": True}
        # else:
        partial = json.loads(s=self._recognizer.PartialResult()).get("partial", "")
        return {"partial": partial, "final": False}

    def is_available(self) -> bool:
        try:
            _ = Model(self.VOSK_MODEL_PATH)
            return True
        except (OSError, RuntimeError, FileNotFoundError) as e:
            logger.error("Vosk model loading failed: %s", e)
            return False


@ASRServiceFactory.register("kyutai")
class KyutaiASRService(ASRServiceBase):
    """
    Kyutai ASR Service implementation.
    Uses the Kyutai (Moshi) Huggingface model for speech recognition.
    Primarly used for the offline version of the app due to its big size.
    """

    def __init__(self):
        pass  # Initialize Kyutai client here

    def transcribe_stream(self, audio_chunk: bytes) -> ASRAnswer:
        # Placeholder implementation
        return {"text": "Kyutai transcription not implemented.", "final": True}
