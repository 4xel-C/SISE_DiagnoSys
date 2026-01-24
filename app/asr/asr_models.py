"""ASR Services implementations using Vosk and Kyutai (Moshi) models."""

import json
import logging
import os
import numpy as np
from typing import Protocol, cast, Type

from vosk import KaldiRecognizer, Model # type: ignore
import sherpa_onnx  # type: ignore

# Use relative imports to avoid circular import through the package
from .base_asr import ASRServiceBase
from .factory import ASRServiceFactory

logger = logging.getLogger(__name__)


ASRService = Type[ASRServiceBase]
ASRAnswer = dict[str, str | bool]

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
            final = True
        else:  # partial result
            result = json.loads(self._recognizer.PartialResult()).get("partial", "")
            logger.debug("Vosk partial result: %s", result)
            final = False
        print("Vosk result:", result)
        return {"text": result, "final": final}

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


# ============================================
# sherpa_onnx
# ============================================

@ASRServiceFactory.register("sherpa_onnx")
class SherpaOnnxASRService(ASRServiceBase):
    """
    Sherpa ONNX ASR Service implementation.
    Uses the Sherpa ONNX model for speech recognition.
    """
    # Paths can be overridden via environment variables
    TOKENS = os.getenv("SHERPA_TOKENS", "data/sherpa-onnx/tokens.txt")
    ENCODER = os.getenv("SHERPA_ENCODER", "data/sherpa-onnx/encoder.onnx")
    DECODER = os.getenv("SHERPA_DECODER", "data/sherpa-onnx/decoder.onnx")
    JOINER = os.getenv("SHERPA_JOINER", "data/sherpa-onnx/joiner.onnx")
    SAMPLE_RATE = 16000

    def __init__(self):
        try:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=self.TOKENS,
                encoder=self.ENCODER,
                decoder=self.DECODER,
                joiner=self.JOINER,
                num_threads=2,
                sample_rate=self.SAMPLE_RATE,
                feature_dim=80,
            )
            self._stream = self._recognizer.create_stream()
            self._available = True
            logger.debug("Sherpa ONNX recognizer initialized with tokens=%s", self.TOKENS)
        except Exception as e:
            logger.error("Failed to initialize Sherpa ONNX recognizer: %s", e)
            self._recognizer = None
            self._stream = None
            self._available = False

    def _pcm16_to_float(self, pcm_bytes: bytes) -> np.ndarray:
        # Accept bytes, bytearray or memoryview. Ensure even length for int16.
        if isinstance(pcm_bytes, memoryview):
            pcm_bytes = pcm_bytes.tobytes()
        elif isinstance(pcm_bytes, bytearray):
            pcm_bytes = bytes(pcm_bytes)

        if not pcm_bytes:
            return np.array([], dtype=np.float32)

        if len(pcm_bytes) % 2 != 0:
            # odd number of bytes -> drop the last dangling byte
            logger.debug("Dropping trailing byte from odd-length PCM buffer (len=%d)", len(pcm_bytes))
            pcm_bytes = pcm_bytes[:-1]

        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe_stream(self, audio_chunk: bytes) -> ASRAnswer:
        if not self._available or self._recognizer is None:
            logger.debug("Sherpa ONNX recognizer not available.")
            return {"text": "", "final": True}

        # empty chunk -> flush final result
        if len(audio_chunk) == 0:
            try:
                self._recognizer.decode_stream(self._stream)
                res = self._recognizer.get_result(self._stream)
                text = getattr(res, "text", str(res) if res else "")
            except Exception as e:
                logger.exception("Error while flushing Sherpa stream: %s", e)
                text = ""

            # reset stream for next utterance
            try:
                self._stream = self._recognizer.create_stream()
            except Exception:
                self._stream = None

            return {"text": text, "final": True}

        # feed audio (expecting PCM16 bytes)
        try:
            audio = self._pcm16_to_float(audio_chunk)
            # API: stream.accept_waveform(sample_rate, np.float32_array)
            self._stream.accept_waveform(self.SAMPLE_RATE, audio)

            # decode while ready and return any available partial result
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
                res = self._recognizer.get_result(self._stream)
                text = getattr(res, "text", str(res) if res else "")
                if text:
                    return {"text": text, "final": False}

        except Exception:
            logger.exception("Error while processing audio chunk for Sherpa ONNX")

        return {"text": "", "final": False}

    def is_available(self) -> bool:
        return bool(self._available)