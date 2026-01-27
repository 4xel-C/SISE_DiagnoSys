# ASR Folder

This folder contains the **Automatic Speech Recognition** (ASR) module of the application. It includes all the necessary components, configurations, and documentation required to implement and manage ASR functionalities.

## Summary
- [Explanations](#explanations): Explanation of the different files and what they do.
- [Model Selection](#model-selection): Explanation of the benchmarks performed to choose the model.
- [Adding a Model](#adding-a-model): How to add a model.

## Explanations

This directory contains three main files: `asr_models.py`, `base_asr.py`, and `factory.py`. Each has a distinct role.

- `base_asr.py` defines the abstract base class `ASRServiceBase` that any ASR model class must inherit from. It provides the interface with two abstract methods:
  - `transcribe_stream(audio_chunk: bytes) -> dict[str, str | bool]`: Processes a chunk of audio for streaming transcription and returns a dictionary with transcription results.
  - `is_available() -> bool`: Checks if the ASR service is available and properly configured.

- `factory.py` handles the model selection. It contains a decorator to easily add classes to the registry and a factory that returns an instance of the chosen class. The `ASRServiceFactory` class uses a registry to store registered services and provides methods to create and check availability of services.

- `asr_models.py` contains the different model implementations. Currently, it includes the `SherpaOnnxASRService` class, which implements ASR using the Sherpa ONNX model for lightweight, online speech recognition suitable for deployment. It processes audio streams and outputs text streams.

The `__init__.py` file imports the main classes and exposes `ASRServiceFactory` and `ASRServiceBase` for use in other parts of the application.

## Model Selection

The model selection was made among 10 ASR models for french language, varying in size and performance. Among the models implemented but not kept:
- Vosk: A lightweight ASR engine, but less accurate for complex audio.
- Kyutai (Moshi): Advanced model, but resource-intensive.
- Whisper: Powerful for various languages, but slower for real-time use.

The chosen model is Sherpa ONNX due to its balance of performance, speed, and suitability for online/streaming transcription.

| Model               | Size     | Accuracy | Speed | Resource Usage | Suitability for Streaming |
|---------------------|----------|----------|-------|----------------|---------------------------|
| Vosk                | ~50 MB   | Medium   | Fast  | Low            | Good                      |
| Kyutai              | ~1.5 GB  | High     | Slow  | High           | Poor                      |
| Whisper             | ~1 GB    | High     | Medium| Medium         | Fair                      |
| Sherpa ONNX         | ~150 MB  | High     | Fast  | Low            | Excellent                 |
| DeepSpeech          | ~1 GB    | High     | Slow  | Medium         | Fair                      |
| Kroko ASR           | ~200 MB  | High     | Fast  | Low            | Good                      |
| NVIDIA FastConformer| ~500 MB  | High     | Fast  | Medium         | Excellent                 |
| faster-whisper      | ~1 GB    | High     | Fast  | Medium         | Good                      |

Benchmarks were performed on a dataset of medical audio samples, evaluating transcription accuracy, latency, and CPU/memory usage. The text streaming capability was a key factor in the decision, as well as the model's ability to run efficiently on limited hardware.

## Adding a Model

To add a model, simply create an adapter class in `asr_models.py` or another file, import the `ASRServiceFactory` from `factory.py`, and add the decorator `@ASRServiceFactory.register(__model_name__)` before the class. This way, it will be available for selection in the factory.
