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
    CHUNK_SAMPLES = 320  # 20 ms @ 16k

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
            logger.debug("Sherpa ONNX initialized")
        except Exception:
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

    def _convert_audio_with_ffmpeg(self, audio_data: bytes) -> bytes:
        """Convert audio data to PCM16 16kHz mono using ffmpeg."""
        print(f"[ASR] Converting audio with FFmpeg, input size: {len(audio_data)} bytes")
        try:
            process = subprocess.Popen(
                self._FFMPEG_CMD,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            pcm_data, stderr = process.communicate(input=audio_data)

            if process.returncode != 0:
                print(f"[ASR] FFmpeg error: {stderr.decode()}")
                logger.error(f"FFmpeg error: {stderr.decode()}")
                return b""

            print(f"[ASR] FFmpeg conversion successful, output size: {len(pcm_data)} bytes")
            return pcm_data
        except Exception as e:
            print(f"[ASR] FFmpeg exception: {e}")
            logger.exception("Failed to convert audio with ffmpeg")
            return b""

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

        # Convert audio to PCM16 16kHz mono
        pcm_data = self._convert_audio_with_ffmpeg(audio_data)
        if not pcm_data:
            print("[ASR] No PCM data after conversion")
            logger.warning("No PCM data after conversion")
            return ""

        # Ensure even length (16-bit samples)
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]

        # Convert to float32
        audio_float = self._pcm16_to_float(pcm_data)
        if audio_float.size == 0:
            print("[ASR] Empty audio after float conversion")
            return ""

        print(f"[ASR] Audio converted to float32, {audio_float.size} samples, duration: {audio_float.size / self.SAMPLE_RATE:.2f}s")

        # Create a new stream for this transcription
        try:
            stream = self._recognizer.create_stream()
            print("[ASR] Sherpa stream created")
        except Exception as e:
            print(f"[ASR] Failed to create stream: {e}")
            logger.exception("Failed to create sherpa stream")
            return ""

        # Feed audio in chunks
        frame_samples = self.CHUNK_SAMPLES
        chunks_fed = 0
        for i in range(0, len(audio_float), frame_samples):
            chunk = audio_float[i : i + frame_samples]
            if len(chunk) > 0:
                try:
                    stream.accept_waveform(self.SAMPLE_RATE, chunk)
                    chunks_fed += 1
                except Exception as e:
                    print(f"[ASR] Error in accept_waveform: {e}")
                    logger.exception("Error in accept_waveform")
                    continue

        print(f"[ASR] Fed {chunks_fed} chunks to recognizer")

        # Signal input finished
        try:
            stream.input_finished()
            print("[ASR] Input finished signaled")
        except Exception as e:
            print(f"[ASR] Error signaling input finished: {e}")
            logger.exception("Error signaling input finished")

        # Decode all frames
        try:
            decode_count = 0
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
                decode_count += 1

            print(f"[ASR] Decoded {decode_count} times")

            result = self._recognizer.get_result(stream)
            text = getattr(result, "text", "").strip()
            print(f"[ASR] Transcription result: '{text}'")
            logger.info(f"Transcription result: {text}")
            return text
        except Exception as e:
            print(f"[ASR] Error during transcription: {e}")
            logger.exception("Error during transcription")
            return ""

    def is_available(self) -> bool:
        """Check if the ASR service is available."""
        return bool(self._available)
