"""ASR Services implementations using Vosk and Kyutai (Moshi) models."""

import logging
import os
import numpy as np
from typing import Type

import sherpa_onnx
import subprocess
import threading
import moshi.models
import torch
import julius
from contextlib import ExitStack

# Use relative imports to avoid circular import through the package
from .base_asr import ASRServiceBase
from .factory import ASRServiceFactory

logger = logging.getLogger(__name__)


ASRService = Type[ASRServiceBase]
ASRAnswer = dict[str, str | bool]


# ============================================
# Kyutai ASR Service
# Big model, prefered to be used offline
# on a local machine with enough resources
# ============================================

@ASRServiceFactory.register("kyutai")
class KyutaiASRService(ASRServiceBase):

    """
    Kyutai (Moshi) ASR service using the `kyutai/stt-1b-en_fr` checkpoint.
    Provides a streaming API similar to the Sherpa implementation: use
    `start_session()`, then repeatedly call `transcribe_stream(bytes)` with
    encoded audio (any format `ffmpeg` understands). Call `end_session()` to
    flush the final result.
    """

    HF_REPO = os.getenv("KYUTAI_HF_REPO", "kyutai/stt-1b-en_fr")

    def __init__(self):
        # session / runtime state
        self._available = False
        self._ffmpeg = None
        self._reader_thread = None
        self._pcm_buffer = bytearray()
        self._last_text = ""
        self._last_final = False
        self._lock = threading.Lock()
        self._session_active = False

        # model objects (loaded from HF)
        self._mimi = None
        self._lm_gen = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._sample_rate = 48000
        self._frame_size = 1600
        self._stream_stack: ExitStack | None = None

        try:
            info = moshi.models.loaders.CheckpointInfo.from_hf_repo(self.HF_REPO)
            self._mimi = info.get_mimi(device=self._device)
            self._tokenizer = info.get_text_tokenizer()

            # choose dtype sensibly: bfloat16 when CUDA available, float32 otherwise
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            lm = info.get_moshi(device=self._device, dtype=dtype)
            self._lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

            # use attributes from mimi if present
            self._sample_rate = getattr(self._mimi, "sample_rate", self._sample_rate)
            self._frame_size = getattr(self._mimi, "frame_size", self._frame_size)

            self._available = True
            logger.debug("Kyutai model loaded from %s on %s", self.HF_REPO, self._device)
        except Exception as e:
            logger.exception("Failed to initialize Kyutai ASR: %s", e)
            self._available = False

    _FFMPEG_CMD_TEMPLATE = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        "{sr}",
        "-f",
        "s16le",
        "-codec:a",
        "pcm_s16le",
        "pipe:1",
    ]

    def _pcm16_to_float(self, pcm_bytes: bytes) -> np.ndarray:
        if isinstance(pcm_bytes, memoryview):
            pcm_bytes = pcm_bytes.tobytes()
        elif isinstance(pcm_bytes, bytearray):
            pcm_bytes = bytes(pcm_bytes)

        if not pcm_bytes:
            return np.array([], dtype=np.float32)

        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _start_ffmpeg_session(self):
        try:
            cmd = [c.format(sr=self._sample_rate) if "{sr}" in c else c for c in self._FFMPEG_CMD_TEMPLATE]
            self._ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception:
            logger.exception("Failed to start ffmpeg for Kyutai")
            self._ffmpeg = None
            return

        self._session_active = True
        self._pcm_buffer = bytearray()
        self._last_text = ""
        self._last_final = False
        self._reader_thread = threading.Thread(target=self._ffmpeg_reader, daemon=True)
        self._reader_thread.start()

    def start_session(self) -> None:
        if not self._available:
            logger.debug("Cannot start Kyutai session: model not available")
            return
        if self._session_active:
            return

        # enter streaming contexts for mimi and lm_gen so .step() can be called
        try:
            self._stream_stack = ExitStack()
            self._stream_stack.enter_context(self._mimi.streaming(1))
            self._stream_stack.enter_context(self._lm_gen.streaming(1))
        except Exception:
            logger.exception("Failed to enter mimi/lm streaming contexts for Kyutai")
            if self._stream_stack is not None:
                try:
                    self._stream_stack.close()
                except Exception:
                    pass
            self._stream_stack = None
            return

        self._start_ffmpeg_session()

    def transcribe_stream(self, audio_chunk: bytes) -> ASRAnswer:
        if not self._available:
            logger.debug("Kyutai model not available")
            return {"text": "", "final": True}

        if not self._session_active:
            logger.debug("transcribe_stream called but no active Kyutai session")
            return {"text": "", "final": False}

        if isinstance(audio_chunk, (bytes, bytearray, memoryview)) and len(audio_chunk) == 0:
            with self._lock:
                return {"text": self._last_text, "final": self._last_final}

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
                    logger.exception("ffmpeg stdin broken pipe while writing chunk (kyutai)")
        except Exception:
            logger.exception("Error while writing to ffmpeg stdin for Kyutai")

        with self._lock:
            return {"text": self._last_text, "final": self._last_final}

    def end_session(self) -> ASRAnswer:
        if not self._session_active:
            with self._lock:
                return {"text": self._last_text, "final": self._last_final}

        try:
            if self._ffmpeg and self._ffmpeg.stdin:
                try:
                    self._ffmpeg.stdin.close()
                except Exception:
                    logger.exception("Error closing ffmpeg stdin during end_session (kyutai)")

            if self._reader_thread:
                self._reader_thread.join(timeout=2.0)
        except Exception:
            logger.exception("Error while ending ffmpeg session (kyutai)")

        with self._lock:
            text = self._last_text
            final = self._last_final

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

        # close streaming contexts if present
        if self._stream_stack is not None:
            try:
                self._stream_stack.close()
            except Exception:
                logger.exception("Error closing mimi/lm streaming contexts for Kyutai")
            finally:
                self._stream_stack = None

        return {"text": text, "final": final}

    def is_available(self) -> bool:
        return bool(self._available)

    def _ffmpeg_reader(self):
        frame_bytes = self._frame_size * 2
        try:
            stdout = self._ffmpeg.stdout if self._ffmpeg else None
            if stdout is None:
                return

            while True:
                data = stdout.read(4096)
                if not data:
                    break

                self._pcm_buffer.extend(data)

                while len(self._pcm_buffer) >= frame_bytes:
                    frame = bytes(self._pcm_buffer[:frame_bytes])
                    del self._pcm_buffer[:frame_bytes]

                    try:
                        audio = self._pcm16_to_float(frame)

                        # convert to torch tensor shaped (1,1,T)
                        audio_tensor = torch.from_numpy(audio).float().to(self._device)
                        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

                        try:
                            audio_tokens = self._mimi.encode(audio_tensor)
                            text_tokens = self._lm_gen.step(audio_tokens)

                            # extract single token id and decode to text piece
                            token_id = int(text_tokens[0, 0, 0].cpu().item())
                            if token_id not in (0, 3):
                                piece = self._tokenizer.id_to_piece(token_id)
                                text = piece.replace("▁", " ")
                                with self._lock:
                                    # append partial tokens to last_text
                                    self._last_text += text
                                    self._last_final = False
                        except Exception:
                            logger.exception("Error running Kyutai encode/LM step")
                    except Exception:
                        logger.exception("Error processing frame in kyutai reader")

            # EOF reached -> mark final
            try:
                with self._lock:
                    self._last_final = True
            except Exception:
                logger.exception("Error flushing kyutai after EOF")

        except Exception:
            logger.exception("FFmpeg reader thread crashed (kyutai)")


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
