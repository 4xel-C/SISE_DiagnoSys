"""ASR Services implementations using Vosk and Kyutai (Moshi) models."""

import logging
import os
import numpy as np
from typing import Type

import sherpa_onnx
import subprocess
import threading

# Use relative imports to avoid circular import through the package
from .base_asr import ASRServiceBase
from .factory import ASRServiceFactory

logger = logging.getLogger(__name__)


ASRService = Type[ASRServiceBase]
ASRAnswer = dict[str, str | bool]

# ============================================
# sherpa_onnx
# lightweight ASR model for online use
# suitable for deployment
# ============================================

@ASRServiceFactory.register("sherpa_onnx")
class SherpaOnnxASRService(ASRServiceBase):
    """
    Sherpa ONNX ASR Service implementation.
    Uses the Sherpa ONNX model for speech recognition.
    """
    # Paths to model files
    _model_path = os.getenv("SHERPA_NCNN_MODEL_PATH", "data/sherpa-onnx")
    TOKENS = f"{_model_path}/tokens.txt"
    ENCODER = f"{_model_path}/encoder.onnx"
    DECODER = f"{_model_path}/decoder.onnx"
    JOINER = f"{_model_path}/joiner.onnx"
    SAMPLE_RATE = 16000

    CHUNK_SAMPLES = 320  # 20 ms frames at 16k
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
            self._stream = self._recognizer.create_stream() # type: ignore
            # session-related attributes (initialized lazily per websocket session)
            self._available = True
            self._ffmpeg = None
            self._reader_thread = None
            self._pcm_buffer = bytearray()
            self._last_text = ""
            self._last_final = False
            self._lock = threading.Lock()
            self._session_active = False

            logger.debug("Sherpa ONNX recognizer initialized with tokens=%s", self.TOKENS)
        except Exception as e:
            logger.error("Failed to initialize Sherpa ONNX recognizer: %s", e)
            self._recognizer = None
            self._stream = None
            self._available = False

    def _pcm16_to_float(self, pcm_bytes: bytes) -> np.ndarray:
        # Accept bytes/bytearray/memoryview and convert to float32 samples
        if isinstance(pcm_bytes, memoryview):
            pcm_bytes = pcm_bytes.tobytes()
        elif isinstance(pcm_bytes, bytearray):
            pcm_bytes = bytes(pcm_bytes)

        if not pcm_bytes:
            return np.array([], dtype=np.float32)

        # length should be multiple of 2; caller ensures frames are full
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _start_ffmpeg_session(self):
        # Start ffmpeg subprocess for this session and reader thread
        try:
            self._ffmpeg = subprocess.Popen(
                self._FFMPEG_CMD,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            logger.exception("Failed to start ffmpeg")
            self._ffmpeg = None
            return

        self._session_active = True
        self._pcm_buffer = bytearray()
        self._last_text = ""
        self._last_final = False
        self._reader_thread = threading.Thread(target=self._ffmpeg_reader, daemon=True)
        self._reader_thread.start()

    def transcribe_stream(self, audio_chunk: bytes) -> ASRAnswer:
        if not self._available or self._recognizer is None:
            logger.debug("Sherpa ONNX recognizer not available.")
            return {"text": "", "final": True}

        # Expect caller to have started a session explicitly via start_session().
        if not self._session_active:
            logger.debug("transcribe_stream called but no active session")
            return {"text": "", "final": False}

        # If caller sends an empty chunk, just return the latest partial (do not close ffmpeg here)
        if isinstance(audio_chunk, (bytes, bytearray, memoryview)) and len(audio_chunk) == 0:
            with self._lock:
                return {"text": self._last_text, "final": self._last_final}

        # Feed encoded chunk to ffmpeg stdin
        try:
            if self._ffmpeg and self._ffmpeg.stdin:
                if isinstance(audio_chunk, memoryview):
                    audio_chunk = audio_chunk.tobytes()
                elif isinstance(audio_chunk, bytearray):
                    audio_chunk = bytes(audio_chunk)

                try:
                    self._ffmpeg.stdin.write(audio_chunk)
                    self._ffmpeg.stdin.flush()
                except BrokenPipeError:
                    logger.exception("ffmpeg stdin broken pipe while writing chunk")
        except Exception:
            logger.exception("Error while writing to ffmpeg stdin for Sherpa ONNX")

        with self._lock:
            return {"text": self._last_text, "final": self._last_final}

    def start_session(self) -> None:
        """Start ffmpeg/decoding session tied to a websocket connection."""
        if not self._available or self._recognizer is None:
            logger.debug("Cannot start session: recognizer not available")
            return
        if self._session_active:
            return
        self._start_ffmpeg_session()

    def end_session(self) -> ASRAnswer:
        """End the active session, flush final result and cleanup."""
        if not self._session_active:
            with self._lock:
                return {"text": self._last_text, "final": self._last_final}

        try:
            if self._ffmpeg and self._ffmpeg.stdin:
                try:
                    self._ffmpeg.stdin.close()
                except Exception:
                    logger.exception("Error closing ffmpeg stdin during end_session")

            if self._reader_thread:
                self._reader_thread.join(timeout=2.0)
        except Exception:
            logger.exception("Error while ending ffmpeg session")

        with self._lock:
            text = self._last_text
            final = self._last_final

        # Cleanup ffmpeg handles
        try:
            if self._ffmpeg:
                try:
                    if self._ffmpeg.stdout:
                        self._ffmpeg.stdout.close()
                except Exception:
                    pass
                try:
                    if self._ffmpeg.stderr:
                        self._ffmpeg.stderr.close()
                except Exception:
                    pass
                try:
                    if self._ffmpeg.poll() is None:
                        self._ffmpeg.kill()
                except Exception:
                    pass
        finally:
            self._ffmpeg = None
            self._reader_thread = None
            self._session_active = False

        return {"text": text, "final": final}

    def is_available(self) -> bool:
        return bool(self._available)

    def _ffmpeg_reader(self):
        # Read raw PCM bytes from ffmpeg.stdout, buffer until we have full frames,
        # convert to float samples and feed into sherpa recognizer.
        frame_bytes = self.CHUNK_SAMPLES * 2
        try:
            stdout = self._ffmpeg.stdout if self._ffmpeg else None
            if stdout is None:
                return

            while True:
                data = stdout.read(4096)
                if not data:
                    break

                # append data to pcm buffer
                self._pcm_buffer.extend(data)

                # while we have at least one full frame, process it
                while len(self._pcm_buffer) >= frame_bytes:
                    frame = bytes(self._pcm_buffer[:frame_bytes])
                    # remove processed bytes
                    del self._pcm_buffer[:frame_bytes]

                    try:
                        audio = self._pcm16_to_float(frame)
                        self._stream.accept_waveform(self.SAMPLE_RATE, audio)
                    except Exception:
                        logger.exception("Error accepting waveform into Sherpa stream")

                    # decode if ready and store partial result
                    try:
                        while self._recognizer.is_ready(self._stream):
                            self._recognizer.decode_stream(self._stream)
                            res = self._recognizer.get_result(self._stream)
                            text = getattr(res, "text", str(res) if res else "")
                            with self._lock:
                                self._last_text = text
                                self._last_final = False
                    except Exception:
                        logger.exception("Error decoding Sherpa stream in reader thread")

            # EOF reached, flush final result
            try:
                if self._recognizer and self._stream:
                    self._recognizer.decode_stream(self._stream)
                    res = self._recognizer.get_result(self._stream)
                    text = getattr(res, "text", str(res) if res else "")
                    with self._lock:
                        self._last_text = text
                        self._last_final = True
            except Exception:
                logger.exception("Error flushing Sherpa stream after EOF")

        except Exception:
            logger.exception("FFmpeg reader thread crashed")


# ============================================
# Note :
# Here could add more ASR services like Kyutai, Vosk, etc.
# Remember to add @ASRServiceFactory.register("service_name") decorator.
# O.B
# ============================================
