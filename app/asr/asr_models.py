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
        self._lock = threading.Lock()
        self._available = False
        self._recognizer = None
        self._stream = None
        self._ffmpeg = None
        self._reader_thread = None
        self._pcm_buffer = bytearray()
        self._last_text: str = ""
        self._last_final: bool = False
        self._session_active = False

        # counters for diagnostics / safety
        self._processed_frame_count: int = (
            0  # number of frames (CHUNK_SAMPLES) fed to recognizer
        )
        self._processed_bytes: int = 0  # number of raw PCM bytes fed to recognizer

        # Trying to initialize sherpa_onnx recognizer
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
        """Convert PCM16 bytes to float32 numpy array.

        Args:
            pcm_bytes (bytes): PCM16 audio data.

        Returns:
            np.ndarray: Float32 numpy array representing the audio signal.
        """
        if isinstance(pcm_bytes, memoryview):
            pcm_bytes = pcm_bytes.tobytes()
        elif isinstance(pcm_bytes, bytearray):
            pcm_bytes = bytes(pcm_bytes)
        if not pcm_bytes:
            return np.array([], dtype=np.float32)

        # length should be multiple of 2; caller ensures frames are full
        return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def _start_ffmpeg_session(self) -> None:
        """Start an ffmpeg subprocess to convert audio to the required format."""
        # if recognizer is not available, do nothing
        if not self._recognizer:
            return

        # start ffmpeg subprocess
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

        # create a fresh recognizer stream for this session
        try:
            self._stream = self._recognizer.create_stream()  # type: ignore
        except Exception:
            logger.exception("Failed to create sherpa stream")
            self._stream = None

        # reset session state
        self._pcm_buffer = bytearray()
        self._last_text = ""
        self._last_final = False
        self._session_active = True

        # reset counters for this session
        self._processed_frame_count = 0
        self._processed_bytes = 0

        # start ffmpeg reader thread
        self._reader_thread = threading.Thread(target=self._ffmpeg_reader, daemon=True)
        self._reader_thread.start()

    def transcribe_stream(self, audio_chunk: bytes) -> ASRAnswer:
        """Transcribe a chunk of audio data in an active session.

        Args:
            audio_chunk (bytes): Audio data chunk to be transcribed.

        Returns:
            ASRAnswer: Transcription result containing text and finality status.
        """
        # if recognizer is not available or session not active, return empty result
        if not self._available or self._recognizer is None:
            return {"text": "", "final": True}
        # if session is not active, return empty result
        if not self._session_active:
            logger.debug("transcribe_stream called without active session")
            return {"text": "", "final": False}

        # ensure audio_chunk is bytes-like
        if isinstance(audio_chunk, memoryview):
            audio_chunk = audio_chunk.tobytes()
        elif isinstance(audio_chunk, bytearray):
            audio_chunk = bytes(audio_chunk)

        # empty chunk => just return last known partial/final
        if not audio_chunk:
            with self._lock:
                return {"text": self._last_text, "final": self._last_final}

        # write audio chunk to ffmpeg stdin
        try:
            # if ffmpeg is not running, return last known result
            if self._ffmpeg and self._ffmpeg.stdin:
                # write audio chunk
                try:
                    self._ffmpeg.stdin.write(audio_chunk)

                    # flush to ensure timely processing
                    self._ffmpeg.stdin.flush()
                # handle broken pipe (ffmpeg closed)
                except BrokenPipeError:
                    logger.exception("BrokenPipe while writing to ffmpeg stdin")
        # catch-all for any other exceptions
        except Exception:
            logger.exception("Error writing to ffmpeg stdin")

        # return last known result
        with self._lock:
            return {"text": self._last_text, "final": self._last_final}

    def start_session(self) -> None:
        """Start a new transcription session."""
        # if recognizer is not available, do nothing
        if not self._available or self._recognizer is None:
            logger.debug("Cannot start session: recognizer unavailable")
            return
        # If session already active, do nothing
        if self._session_active:
            return
        # else start new session
        self._start_ffmpeg_session()

    def end_session(self) -> ASRAnswer:
        """End session and return final transcription."""
        # if session not active, return last known result
        if not self._session_active:
            with self._lock:
                return {"text": self._last_text, "final": self._last_final}

        # 1. Stop accepting new audio
        self._session_active = False

        # 2. Close ffmpeg stdin to signal EOF
        try:
            logger.debug(
                "self._ffmpeg and self._ffmpeg.stdin %s",
                self._ffmpeg and self._ffmpeg.stdin,
            )
            if self._ffmpeg and self._ffmpeg.stdin:
                self._ffmpeg.stdin.close()
        except Exception:
            logger.exception("Error closing ffmpeg stdin")

        # 3. Wait for reader thread to process remaining data
        reader = self._reader_thread
        if reader and reader.is_alive():
            reader.join(timeout=5.0)

        # 4. If reader already finalized, return that result
        with self._lock:
            if self._last_final:
                self._cleanup()
                return {"text": self._last_text, "final": True}

        # 5. Manual flush if reader didn't finalize
        try:
            if self._stream and self._recognizer:
                # Signal input is finished
                self._stream.input_finished()

                # Decode remaining frames
                while self._recognizer.is_ready(self._stream):
                    self._recognizer.decode_stream(self._stream)

                # Get final result
                res = self._recognizer.get_result(self._stream)
                final_text = getattr(res, "text", "").strip()

                with self._lock:
                    if final_text:
                        self._last_text = final_text
                    self._last_final = True
        except Exception:
            logger.exception("Error during manual flush")

        # 6. Cleanup and return
        with self._lock:
            result = {"text": self._last_text, "final": True}

        self._cleanup()
        return result

    def _cleanup(self) -> None:
        """Clean up resources after session ends."""
        try:
            if self._ffmpeg:
                for stream in (self._ffmpeg.stdout, self._ffmpeg.stderr):
                    if stream:
                        try:
                            stream.close()
                        except Exception:
                            pass
                if self._ffmpeg.poll() is None:
                    self._ffmpeg.kill()
            self._ffmpeg = None
            self._reader_thread = None
            self._stream = None
        except Exception:
            logger.exception("Error during cleanup")

    def is_available(self) -> bool:
        """Check if the ASR service is available."""
        return bool(self._available)

    def _ffmpeg_reader(self) -> None:
        """Reader thread to read PCM data from ffmpeg stdout,"""
        # if ffmpeg is not running, exit
        if self._ffmpeg is None or self._ffmpeg.stdout is None:
            return

        # number of bytes per CHUNK_SAMPLES frame
        frame_bytes = self.CHUNK_SAMPLES * 2
        stdout = self._ffmpeg.stdout

        # main read loop
        try:
            while True:
                # read data from ffmpeg stdout
                data = stdout.read(4096)
                if not data:
                    break

                # accept only bytes-like, keep 16-bit alignment
                if not isinstance(data, (bytes, bytearray, memoryview)):
                    logger.warning("ffmpeg reader: unexpected data type %s", type(data))
                    continue
                # convert memoryview to bytes
                if isinstance(data, memoryview):
                    data = data.tobytes()

                # append data to pcm buffer
                self._pcm_buffer.extend(data)

                # alignment check: must be even (16-bit samples)
                if len(self._pcm_buffer) % 2 != 0:
                    logger.debug(
                        "Dropping odd trailing byte from pcm buffer (len=%d)",
                        len(self._pcm_buffer),
                    )
                    self._pcm_buffer = self._pcm_buffer[:-1]

                # process full CHUNK_SAMPLES frames only
                while len(self._pcm_buffer) >= frame_bytes:
                    frame = bytes(self._pcm_buffer[:frame_bytes])
                    del self._pcm_buffer[:frame_bytes]

                    # quick validation of frame content/size before handing to model
                    if not frame or len(frame) != frame_bytes:
                        logger.warning(
                            "Skipping malformed frame (len=%d expected=%d)",
                            len(frame),
                            frame_bytes,
                        )
                        continue

                    try:
                        # convert pcm16 frame to float32 numpy array
                        audio = self._pcm16_to_float(frame)
                        if audio.size != self.CHUNK_SAMPLES:
                            logger.warning(
                                "Skipping frame with unexpected sample count %d (expected=%d)",
                                audio.size,
                                self.CHUNK_SAMPLES,
                            )
                            continue

                        # feed audio frame to sherpa stream
                        if self._stream:
                            try:
                                self._stream.accept_waveform(self.SAMPLE_RATE, audio)  # type: ignore
                            except Exception:
                                logger.exception(
                                    "accept_waveform failed; skipping frame"
                                )
                                continue

                        # update counters
                        self._processed_frame_count += 1
                        self._processed_bytes += frame_bytes

                    except Exception:
                        logger.exception(
                            "Error converting pcm frame to float; skipping frame"
                        )
                        continue

                    # decode as long as recognizer is ready; defend against sherpa native GetFrames issues
                    try:
                        while (
                            self._recognizer
                            and self._stream
                            and self._recognizer.is_ready(self._stream)
                        ):
                            self._recognizer.decode_stream(self._stream)
                            res = self._recognizer.get_result(self._stream)
                            text = getattr(res, "text", str(res) if res else "")
                            with self._lock:
                                self._last_text = text
                                self._last_final = False
                    # handle sherpa exceptions gracefully
                    except Exception as e:
                        msg = str(e)
                        if "GetFrames" in msg:
                            logger.warning(
                                "Sherpa reported GetFrames issue: %s | frames=%d bytes=%d pcm_remain=%d",
                                msg,
                                self._processed_frame_count,
                                self._processed_bytes,
                                len(self._pcm_buffer),
                            )
                            with self._lock:
                                self._last_final = True
                            return
                        logger.exception("Error decoding sherpa stream")

            # EOF reached: flush final result
            try:
                if self._recognizer and self._stream:
                    # Signal that input is finished
                    self._stream.input_finished()

                    # Decode remaining frames
                    while self._recognizer.is_ready(self._stream):
                        self._recognizer.decode_stream(self._stream)

                    # Get final result
                    res = self._recognizer.get_result(self._stream)
                    text = getattr(res, "text", "").strip()
                    with self._lock:
                        if text:
                            self._last_text = text
                        self._last_final = True
                    logger.info("Reader finalized: %s", text)
            except Exception:
                logger.exception("Error flushing final result")
        except Exception:
            logger.exception("FFmpeg reader crashed")


# ============================================
# Note :
# Here could add more ASR services like Kyutai, Vosk, etc.
# Remember to add @ASRServiceFactory.register("service_name") decorator.
# and to import them in app/asr/__init__.py
# O.B
# ============================================
