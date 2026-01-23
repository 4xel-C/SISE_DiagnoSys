"""ASR Services implementations using Vosk and Kyutai (Moshi) models."""

import json
import logging
import os
from typing import Protocol, cast

from vosk import KaldiRecognizer, Model # type: ignore

from app.asr.base import ASRServiceBase
from app.asr.factory import ASRServiceFactory

logger = logging.getLogger(__name__)


type ASRService = type[ASRServiceBase]
type ASRAnswer = dict[str, str | bool]

# ============================================
# Vosk ASR Service
# ============================================


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
            return {"text": "", "final": False}

        # if accepted as final
        if self._recognizer.AcceptWaveform(audio_chunk):
            result = json.loads(self._recognizer.Result()).get("text", "")
            logger.debug("Vosk final result: %s", result)
        else: # partial result
            result = json.loads(s=self._recognizer.PartialResult()).get("partial", "")
            logger.debug("Vosk partial result: %s", result)
        print("Vosk result:", result)
        return {"text": result, "final": False}

    def is_available(self) -> bool:
        try:
            _ = Model(self.VOSK_MODEL_PATH)
            return True
        except (OSError, RuntimeError, FileNotFoundError) as e:
            logger.error("Vosk model loading failed: %s", e)
            return False


# ============================================
# Kyutai ASR Service
# ============================================


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
