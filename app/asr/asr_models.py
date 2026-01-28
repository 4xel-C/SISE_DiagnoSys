"""ASR Services implementations using Sherpa ONNX model."""

import logging
import os
import subprocess

import numpy as np
import sherpa_onnx

from .base_asr import ASRServiceBase
from .factory import ASRServiceFactory

logger = logging.getLogger(__name__)


@ASRServiceFactory.register("sherpa_onnx")
class SherpaOnnxASRService(ASRServiceBase):
    """
    Sherpa ONNX ASR Service implementation.
    Transcribes complete audio data (non-streaming).
    """

    _model_path = os.getenv("SHERPA_NCNN_MODEL_PATH", "data/sherpa-onnx")
    TOKENS = f"{_model_path}/tokens.txt"
    ENCODER = f"{_model_path}/encoder.onnx"
    DECODER = f"{_model_path}/decoder.onnx"
    JOINER = f"{_model_path}/joiner.onnx"
    SAMPLE_RATE = 16000

    _FFMPEG_CMD = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-codec:a",
        "pcm_s16le",
        "pipe:1",
    ]

    def __init__(self) -> None:
        self._available = False
        self._recognizer = None

        print(f"[ASR] Initializing Sherpa ONNX...")
        print(f"[ASR] Model path: {self._model_path}")
        print(f"[ASR] Tokens: {self.TOKENS}")
        print(f"[ASR] Encoder: {self.ENCODER}")

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
            self._available = True
            print("[ASR] Sherpa ONNX initialized successfully")
            logger.debug("Sherpa ONNX initialized")
        except Exception as e:
            print(f"[ASR] Failed to initialize Sherpa ONNX: {e}")
            logger.exception("Failed to initialize Sherpa ONNX")
            self._recognizer = None
            self._available = False

    def _pcm16_to_float(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert PCM16 bytes to float32 numpy array."""
        if isinstance(pcm_bytes, memoryview):
            pcm_bytes = pcm_bytes.tobytes()
        elif isinstance(pcm_bytes, bytearray):
            pcm_bytes = bytes(pcm_bytes)
        if not pcm_bytes:
            return np.array([], dtype=np.float32)
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe complete audio data.

        Args:
            audio_data: Complete audio bytes (WebM/Opus format).

        Returns:
            str: The transcription text.
        """
        print(f"[ASR] transcribe() called with {len(audio_data) if audio_data else 0} bytes")

        if not self._available or self._recognizer is None:
            print("[ASR] Service not available!")
            logger.warning("ASR service not available")
            return ""

        if not audio_data:
            print("[ASR] No audio data received")
            return ""

        # Convert audio with FFmpeg
        try:
            ffmpeg = subprocess.Popen(
                self._FFMPEG_CMD,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            pcm_data, stderr = ffmpeg.communicate(input=audio_data)

            if ffmpeg.returncode != 0:
                print(f"[ASR] FFmpeg error: {stderr.decode()}")
                return ""

            print(f"[ASR] FFmpeg converted {len(audio_data)} -> {len(pcm_data)} bytes")
        except Exception as e:
            print(f"[ASR] FFmpeg failed: {e}")
            return ""

        # Ensure even length (16-bit samples)
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        # Convert to float32
        audio_float = self._pcm16_to_float(pcm_data)
        if audio_float.size == 0:
            print("[ASR] Empty audio after conversion")
            return ""

        print(f"[ASR] Audio: {audio_float.size} samples, {audio_float.size / self.SAMPLE_RATE:.2f}s")

        # Create stream and feed complete audio
        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(self.SAMPLE_RATE, audio_float)
            stream.input_finished()
            print("[ASR] Audio fed to recognizer")
        except Exception as e:
            print(f"[ASR] Error feeding audio: {e}")
            return ""

        # Decode
        try:
            decode_count = 0
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
                decode_count += 1

            print(f"[ASR] Decoded {decode_count} times")

            result = self._recognizer.get_result(stream)
            text = result.strip() if isinstance(result, str) else getattr(result, "text", str(result)).strip()
            text = text.lower()
            print(f"[ASR] Transcription result: '{text}'")
            return text
        except Exception as e:
            print(f"[ASR] Error during decode: {e}")
            return ""

    def is_available(self) -> bool:
        """Check if the ASR service is available."""
        return bool(self._available)
