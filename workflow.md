# Merci chatGPT ❤️

Here’s one practical architecture that keeps everything local and uses **faster-whisper** for near real-time French transcription. The idea is to stream Opus chunks from the browser, decode them to PCM as they arrive, feed short buffers (e.g., 2–5 seconds) to faster-whisper, and emit partial transcripts.

---

## 1. Frontend (already in place)

You already stream audio chunks over WebSocket. Keep your `MediaRecorder` chunk size ~250 ms to 500 ms.

---

## 2. Flask server: receive & queue chunks

Use a queue to hand off raw Opus/WebM chunks (binary) to a background worker. The worker will decode and run the model.

```python
# server/audio_stream.py
import io
import queue
import threading
from flask_sock import Sock

sock = Sock()
opus_queue = queue.Queue()
stop_event = threading.Event()

@sock.route("/ajax/audio")
def audio(ws):
    try:
        while True:
            data = ws.receive()
            if data is None:
                break
            opus_queue.put(data)
    finally:
        opus_queue.put(None)  # signal stream end
```

Initialize `sock` in `create_app()` as described earlier.

---

## 3. Decoding Opus chunks to PCM

Use **PyAV** or **ffmpeg** as a subprocess. Here’s a PyAV-based decoder that yields PCM frames (16 kHz mono). Install `av` (PyAV) and `soundfile` (or `numpy`) for audio manipulation.

```python
# server/decoder.py
import av
import numpy as np

def decode_webm_chunks(chunks):
    """
    chunks: iterable of bytes (WebM/Opus)
    yields numpy arrays of float32 PCM at 16 kHz mono
    """
    container = av.open(io.BytesIO(), format="webm", mode="r",
                        buffer_size=4 * 1024)
    stream = container.streams.audio[0]

    for chunk in chunks:
        if chunk is None:
            break
        container.mux(av.Packet.from_bytes(chunk))
        for frame in container.decode(stream):
            pcm = frame.to_ndarray(format="s16").astype("float32")
            pcm /= 32768.0
            yield pcm.reshape(-1)
```

(Alternatively run `ffmpeg` in `-i pipe:0 -ar 16000 -ac 1 -f s16le pipe:1` mode and feed chunks into stdin.)

---

## 4. Buffering for faster-whisper

We collect ~2–5 seconds of audio (~32k–80k samples at 16 kHz) before calling the model. Smaller buffers reduce latency but hurt accuracy. Use overlap (e.g., keep the last 0.5 s) to avoid cutting words mid-line.

```python
# server/buffer.py
import numpy as np
from collections import deque

class SlidingBuffer:
    def __init__(self, chunk_samples=32000, overlap_samples=8000):
        self.chunk_samples = chunk_samples
        self.overlap_samples = overlap_samples
        self.buffer = deque()
        self.total_samples = 0

    def add_samples(self, samples: np.ndarray):
        self.buffer.append(samples)
        self.total_samples += len(samples)

    def ready(self):
        return self.total_samples >= self.chunk_samples

    def pop_chunk(self):
        # gather chunk_samples + overlap
        needed = self.chunk_samples
        collected = []
        while needed > 0 and self.buffer:
            block = self.buffer.popleft()
            collected.append(block)
            needed -= len(block)
        data = np.concatenate(collected)

        # keep overlap portion for context
        overlap = data[-self.overlap_samples:]
        self.buffer.clear()
        self.buffer.append(overlap)
        self.total_samples = len(overlap)

        return data
```

---

## 5. faster-whisper worker

Install `faster-whisper` (GPU recommended): `pip install faster-whisper`.

```python
# server/transcriber.py
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",  # choose small/medium for speed/accuracy tradeoff
    device="cuda",  # or "cpu"
    compute_type="float16"
)

def transcribe_chunk(audio_samples: np.ndarray, prompt: str = ""):
    segments, _ = model.transcribe(
        audio_samples,
        language="fr",
        vad_filter=True,
        initial_prompt=prompt,
        beam_size=5,
        temperature=0.0
    )
    text = "".join(segment.text for segment in segments).strip()
    return text
```

---

## 6. Putting it together

Spin up a worker thread once your app initializes:

```python
# server/worker.py
import threading
from .decoder import decode_webm_chunks
from .buffer import SlidingBuffer
from .transcriber import transcribe_chunk
from .audio_stream import opus_queue, stop_event

def transcription_worker(on_text_callback):
    decoder = decode_webm_chunks(iter(lambda: opus_queue.get(), None))
    buffer = SlidingBuffer(chunk_samples=32000, overlap_samples=8000)
    prompt = ""

    for pcm in decoder:
        buffer.add_samples(pcm)
        if buffer.ready():
            chunk = buffer.pop_chunk()
            text = transcribe_chunk(chunk, prompt=prompt)
            if text:
                prompt = (prompt + " " + text)[-200:]  # keep recent context
                on_text_callback(text)

def start_worker(app):
    def callback(text):
        with app.app_context():
            print("Partial:", text)
            # TODO: push to websocket, SSE, etc.

    thread = threading.Thread(target=transcription_worker, args=(callback,), daemon=True)
    thread.start()
```

Call `start_worker(app)` from `create_app()` after `sock.init_app(app)`.

---

## 7. Emitting results to the browser

You can broadcast partial transcripts back via another WebSocket or Server-Sent Events. For a simple bidirectional channel, reuse the same WebSocket: keep a reference to the client connection (`ws`) and send JSON like `{"partial": "text"}` whenever `on_text_callback` fires. Or have a separate endpoint for clients to subscribe to transcripts.

---

## Key tuning knobs

1. **Chunk size (latency vs accuracy)**: 2–3 seconds gives decent French accuracy with Whisper-small/medium.
2. **Overlap**: 0.5–1.0 seconds prevents truncating words.
3. **Prompt management**: Keep a rolling context (200–300 chars) to help the model continue sentences without repeating.
4. **Model size**: `small` or `medium` typically run real-time on a mid-range GPU. On CPU, use `base` or `small.en` equivalents (though slower).
5. **VAD**: `vad_filter=True` reduces silence processing; you can also add your own VAD (e.g., Silero) to skip quiet segments.

---

### Summary

- Stream audio via WebSocket → queue.
- Decode Opus to PCM incrementally.
- Buffer ~2–5 s of samples with overlap.
- Call `faster-whisper` on each buffer; pass previous transcript as prompt.
- Emit partial transcripts back to the client.

This gives “near real-time” behavior with latency roughly equal to chunk size + model inference time, while staying fully self-hosted.