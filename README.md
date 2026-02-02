<img width="1395" height="569" alt="image" src="https://github.com/user-attachments/assets/7160ecf4-510b-463d-89f5-d51a68eea35a" />

# **AI-Powered Medical Diagnostic Support System**

[![Deployment](https://img.shields.io/badge/deployment-live-brightgreen)](https://diagnosys.cyrizon.fr/)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.1.2-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live Demo:** [https://diagnosys.cyrizon.fr/](https://diagnosys.cyrizon.fr/)

DiagnoSys is a web application designed for healthcare professionals, combining patient record management, real-time speech recognition, and a Retrieval-Augmented Generation (RAG) pipeline to assist clinical decision-making.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Problem We Solve](#the-problem-we-solve)
3. [How It Works](#how-it-works)
4. [Features](#features)
5. [Architecture](#architecture)
6. [Tech Stack](#tech-stack)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Running the Application](#running-the-application)
10. [Project Structure](#project-structure)
11. [API Reference](#api-reference)
12. [Security](#security)
13. [Environmental Impact Measurement](#environmental-impact-measurement)
14. [Development Guidelines](#development-guidelines)

---

## Project Overview

DiagnoSys is a medical diagnostic assistance solution designed for French-speaking clinical environments. The application enables practitioners to:

- **Capture patient information** via text input or real-time voice dictation
- **Obtain diagnostic suggestions** based on medical context and a document knowledge base
- **Train** through a simulation mode where the LLM embodies a virtual patient
- **Monitor environmental impact** of language model API calls

The system integrates a RAG (Retrieval-Augmented Generation) pipeline that combines patient information with relevant medical documents to generate ranked diagnostic hypotheses.

---

## The Problem We Solve

### Emergency Room Reality

Emergency departments face critical challenges that directly impact patient care:

- **Time pressure**: Physicians must make rapid decisions with incomplete information. In France, emergency rooms handle over 21 million visits annually, with average wait times that can exceed several hours during peak periods.

- **Cognitive overload**: A single emergency physician may juggle 10-15 patients simultaneously, each with different symptoms, histories, and urgency levels. This mental load increases the risk of diagnostic errors.

- **Information fragmentation**: Patient history, current symptoms, vital signs, and relevant medical literature are scattered across different systems, making it difficult to synthesize a complete clinical picture.

- **Documentation burden**: Physicians spend a significant portion of their time on administrative tasks and documentation rather than direct patient care.

### How DiagnoSys Helps

DiagnoSys acts as a **cognitive assistant** for emergency physicians, not replacing their expertise but augmenting their capabilities:

| Challenge | DiagnoSys Solution |
|-----------|-------------------|
| **Time pressure** | Voice-based input allows hands-free documentation while examining patients. Automatic context updates every 20 seconds ensure nothing is lost. |
| **Cognitive overload** | AI-generated diagnostic hypotheses serve as a "second opinion", helping physicians consider differential diagnoses they might overlook under pressure. |
| **Information fragmentation** | The RAG pipeline automatically retrieves relevant medical literature and similar past cases, bringing all pertinent information into one view. |
| **Documentation burden** | Natural language processing transforms spoken observations into structured medical context, reducing manual data entry. |

### Design Philosophy

DiagnoSys is built on the principle that **AI should assist, not replace**. The system:

- Presents diagnostic **suggestions**, not decisions — the physician always has the final word
- Prioritizes **transparency** by showing which documents and similar cases informed each suggestion
- Works **offline-first** for speech recognition, ensuring reliability even with unstable hospital networks
- Tracks **environmental impact** because responsible AI deployment matters in healthcare

### How to Use the Application

DiagnoSys offers two methods to update a patient's medical context. Both methods trigger the full RAG pipeline: document retrieval, similar case identification, and diagnostic generation.

#### Text Input Mode

1. **Select a patient** from the list or create a new one
2. **Edit the context** directly in the rich text editor (Quill)
3. **Click "Update Context"** to save your modifications
4. The system automatically:
   - Searches for the most relevant medical documents
   - Identifies similar patient cases
   - Generates 3 ranked diagnostic hypotheses
   - Updates the "Related Documents" and "Similar Cases" panels

#### Voice Input Mode

1. **Select a patient** from the list or create a new one
2. **Press the microphone button** (or use Shift+Space) to start recording
3. **Speak naturally** — the audio is captured in real-time with visual waveform feedback
4. **Automatic processing every 20 seconds**: audio is transcribed locally, validated by the security guardrail, then processed by the LLM to **intelligently synthesize** new information into the existing context
5. **Stop recording** when finished — the RAG pipeline triggers automatically

> **Note**: The voice mode uses LLM-based context synthesis, meaning the AI doesn't simply append the transcription but intelligently integrates new information (symptoms, observations, medical history) into a coherent and structured medical context.

#### Simulation Mode (Chatbot)

1. **Open the chat interface** to practice with a virtual patient
2. The LLM embodies a patient based on an existing profile
3. Ask questions as you would in a real consultation — the simulated patient responds naturally

---

## How It Works

### Workflow Overview

The application follows a multi-stage pipeline:

1. **Data Capture**: Healthcare professionals input patient information via text or voice recording
2. **Speech-to-Text**: Audio is transcribed locally using Sherpa ONNX (no cloud dependency)
3. **Security Validation**: Input passes through a ML-based guardrail to detect prompt injection attempts
4. **Context Synthesis**: The LLM summarizes and integrates new information into the patient's medical context
5. **Vector Embedding**: Patient context is embedded and stored in ChromaDB for semantic search
6. **Document Retrieval**: The system retrieves the top 5 most relevant medical documents
7. **Diagnostic Generation**: The LLM generates 3 diagnostic hypotheses ranked by probability

### Audio Recording Flow

- Audio is captured via the browser's MediaRecorder API (WebM/Opus format)
- Chunks are accumulated every 250ms
- **Automatic periodic send every 20 seconds** without stopping the recording
- Final audio sent when recording stops
- Transcription and context update happen server-side

---

## Features

### Patient Management

- **Create and update** patient records with complete medical information
- **Record vital signs**: heart rate, respiratory rate, O2 saturation, blood pressure, temperature
- **Severity classification** using triage color codes (grey, green, yellow, red)
- **Semantic search** for similar patient cases via vector embeddings

### Diagnostic Assistance (RAG)

The RAG pipeline orchestrates multiple steps to generate diagnostic hypotheses:

1. **Context update**: integration of new information (text or transcribed audio)
2. **Document retrieval**: search for the 5 most relevant medical documents in the vector database
3. **Similar case retrieval**: identification of patients with similar profiles
4. **Diagnostic generation**: the LLM produces 3 hypotheses ranked by decreasing probability

<img width="1913" height="905" alt="image" src="https://github.com/user-attachments/assets/41b55378-cfec-462f-a4a9-145fb37ef5e3" />


### Real-Time Speech Recognition

- **Audio recording** via Web MediaRecorder API (WebM/Opus format)
- **Real-time waveform visualization** on HTML5 canvas
- **Automatic periodic sending** every 20 seconds without recording interruption
- **Offline transcription** via Sherpa ONNX (no cloud dependency)
- **Seamless integration**: patient context is automatically enriched

### Patient Simulation (Chatbot)

Training mode for practitioners:

- The LLM embodies a virtual patient based on an existing profile
- Natural and realistic responses to the doctor's questions
- Conversation history maintained per session
- The patient doesn't reveal all symptoms at once, simulating a real consultation

<img width="484" height="911" alt="image" src="https://github.com/user-attachments/assets/bd039014-8e99-4f20-b6bd-b1b06300444a" />


### Metrics and Environmental Impact

Complete tracking of language model usage:

- **Tokens**: input, output, total per request and per day
- **Cost**: calculation in USD based on Mistral pricing
- **Latency**: average response time
- **Environmental impact** via EcoLogits:
  - Energy consumption (kWh)
  - Carbon footprint (kg CO2 eq)
  - Abiotic depletion (mg Sb eq)
  - Primary energy demand (MJ)
  - Water consumption (liters)

<img width="1571" height="827" alt="image" src="https://github.com/user-attachments/assets/c93b4640-64b5-44d5-98ba-a9ddaefb4dec" />


---

## Architecture

<img width="2642" height="1528" alt="schema_architecture" src="https://github.com/user-attachments/assets/dbaf77a0-8cbf-45f1-9ca8-2e97c6fac4f0" />


---

## Tech Stack

| Component | Technology | Version | Role |
|-----------|------------|---------|------|
| **Backend** | Flask | 3.1.2 | WSGI web framework |
| **ORM** | SQLAlchemy | 2.0.45 | Object-relational mapping |
| **Relational DB** | SQLite | - | Patients, documents, metrics storage |
| **Vector DB** | ChromaDB | 1.4.1 | Embeddings storage, semantic search |
| **Embeddings** | Sentence-Transformers | 5.2.0 | Vector generation (all-MiniLM-L6-v2) |
| **LLM** | Mistral AI | 1.10.1 | Text generation via API |
| **ASR** | Sherpa ONNX | 1.12.23 | Offline speech transcription |
| **Validation** | Pydantic | 2.12.5 | Data schemas |
| **Migrations** | Alembic | 1.18.1 | Database versioning |
| **WSGI Server** | Gunicorn | 21.2.0 | Production server |
| **Env. Impact** | EcoLogits | 0.9.2 | LLM carbon footprint measurement |
| **Frontend** | Vanilla JS | ES6+ | User interface |
| **Rich Editor** | Quill | 2.0.3 | Patient context editing |
| **Charts** | Plotly | 3.1.0 | Statistics dashboard |

---

## Installation

### Prerequisites

- **Python 3.13+**
- **FFmpeg** (WebM to PCM audio conversion)
- **Mistral API Key** (https://console.mistral.ai/)

### Local Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd SISE_DiagnoSys

# 2. Create Python environment with UV
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
uv sync

# 4. Configure environment variables
cp .env.exemple .env
# Edit .env with your Mistral API key
```

### Docker Installation

```bash
# Build and launch with Docker Compose
docker-compose up --build

# Available services:
# - Application: http://localhost:8000
# - ChromaDB: http://localhost:8001
```

---

## Configuration

### Environment Variables (.env)

```env
# Application
APP_NAME=diagnosys

# Logging
LOG_LEVEL=DEBUG                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
PRINT_CONSOLE_LOGS=false

# Database
DATABASE_PATH=data/diagnosys.db
CHROMA_DB_PATH=data/chroma

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=mistral-small-latest     # See MODELS in llm_options.py

# API
MISTRAL_API_KEY=<your-api-key>

# ASR
SHERPA_NCNN_MODEL_PATH=data/sherpa-onnx

# Mode
ONLINE_MODE=1                      # 1=online (API), 0=offline
```

### Available Mistral Models

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| ministral-3b-latest | 0.04 | 0.04 |
| ministral-8b-latest | 0.10 | 0.10 |
| mistral-small-latest **(recommended)**| 0.20 | 0.60 |
| mistral-medium-latest | 2.50 | 7.50 |
| mistral-large-latest | 2.00 | 6.00 |
| codestral-latest | 0.30 | 0.90 |

---

## Running the Application

### Development Mode

```bash
python run.py
# Available at http://127.0.0.1:8000
```

### Production Mode

```bash
gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
         -b 0.0.0.0:8000 \
         app:app
```

---

## Project Structure

```
SISE_DiagnoSys/
├── app/
│   ├── __init__.py              # Flask factory
│   ├── init.py                  # Configuration and services
│   ├── routes.py                # HTML page routes
│   ├── ajax.py                  # AJAX endpoints (internal API)
│   │
│   ├── config/                  # Configuration
│   │   ├── database.py          # SQLite/SQLAlchemy connection
│   │   ├── logging_config.py    # Logging configuration
│   │   └── vector_db.py         # ChromaDB client
│   │
│   ├── models/                  # SQLAlchemy models (ORM)
│   │   ├── patient.py           # Patients table
│   │   ├── document.py          # Medical documents table
│   │   └── llm_usage.py         # LLM metrics table
│   │
│   ├── schemas/                 # Pydantic schemas (validation)
│   │   ├── patient_schema.py
│   │   ├── document_schema.py
│   │   └── llm_metrics_schema.py
│   │
│   ├── services/                # Business logic
│   │   ├── patient_service.py   # Patient CRUD + embeddings
│   │   ├── document_service.py  # Document CRUD
│   │   ├── rag_service.py       # RAG pipeline orchestration
│   │   ├── chat_service.py      # Patient simulation
│   │   └── llm_usage_service.py # Metrics tracking
│   │
│   ├── rag/                     # RAG components
│   │   ├── llm.py               # Mistral API handler
│   │   ├── llm_options.py       # Configs, prompts, templates
│   │   ├── vector_store.py      # ChromaDB operations
│   │   ├── vectorizer.py        # Embeddings generation
│   │   └── guardrail.py         # Prompt injection detection
│   │
│   ├── asr/                     # Speech recognition
│   │   ├── base_asr.py          # Abstract interface
│   │   ├── asr_models.py        # Sherpa ONNX implementation
│   │   └── factory.py           # Factory pattern
│   │
│   ├── scraper/                 # Medical document scraping
│   │   └── nlm_sp_scraper.py    # PubMed/NLM scraper
│   │
│   ├── pipelines/               # Processing pipelines
│   │   └── document_loader/     # Document loading
│   │
│   ├── static/                  # Frontend assets
│   │   ├── js/
│   │   │   ├── audio-record.js  # Audio recording + waveform
│   │   │   ├── chat.js          # Chat interface
│   │   │   ├── patient.js       # Patient profile
│   │   │   └── modules/
│   │   │       ├── streamer.js  # Audio stream management
│   │   │       └── loader.js    # AJAX loading
│   │   ├── css/                 # Styles
│   │   └── images/              # SVG icons
│   │
│   └── templates/               # Jinja2 templates
│       ├── index.html           # SPA shell
│       ├── patient.html         # Diagnostic view
│       ├── chat.html            # Chat interface
│       └── elements/            # Reusable components
│
├── data/                        # Data (gitignored)
│   ├── diagnosys.db             # SQLite database
│   ├── chroma/                  # Vector database
│   ├── sherpa-onnx/             # ASR model
│   └── ml_models/               # Guardrail model
│
├── logs/                        # Application logs
├── alembic/                     # DB migrations
├── docker-compose.yml           # Docker config
├── Dockerfile                   # Docker image
├── pyproject.toml               # Python dependencies
├── uv.lock                      # UV lock file
└── run.py                       # Dev entry point
```

---

## API Reference

### AJAX Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ajax/audio_stt/<patient_id>` | Audio transcription and context update |
| `POST` | `/ajax/process_rag/<patient_id>` | RAG diagnostic generation |
| `POST` | `/ajax/load_agent/<patient_id>` | Initialize chat session (simulation) |
| `POST` | `/ajax/query_agent/<patient_id>` | Send message to simulated patient |
| `POST` | `/ajax/update_context/<patient_id>` | Manual context update |
| `GET` | `/ajax/search_patients?query=...` | Search patients by name |
| `GET` | `/ajax/get_profile/<patient_id>` | Patient metadata |
| `GET` | `/ajax/get_context/<patient_id>` | Patient medical context |
| `GET` | `/ajax/get_diagnostic/<patient_id>` | Generated diagnostic |
| `GET` | `/ajax/get_related_documents/<patient_id>` | Similar documents |
| `GET` | `/ajax/get_related_cases/<patient_id>` | Similar patient cases |
| `GET` | `/ajax/render_patient/<patient_id>` | Patient HTML template |
| `GET` | `/ajax/render_chat` | Chat HTML template |

---

## Security

### Prompt Injection Detection (Guardrail)

The system integrates an ML classifier to detect prompt injection attempts at **3 critical checkpoints**:

| Checkpoint | Location | Purpose |
|------------|----------|---------|
| `raw_transcript` | After audio transcription | Validate raw transcribed text |
| `synthesized_context` | After LLM summarization | Validate synthesized context |
| `pre_diagnosis` | Before diagnostic LLM call | Final validation before diagnosis |

**Classifier characteristics:**
- Ensemble model trained on jailbreak/injection datasets
- Features: embeddings (paraphrase-multilingual-MiniLM-L12-v2) + handcrafted features
- Suspicious pattern detection (regex): "ignore previous", "pretend you are", etc.
- Configurable threshold (default: 0.5)

### Other Security Measures

- **Sensitive variables**: API keys via `.env` (never hardcoded)
- **Type annotations**: complete typing for static validation
- **Pydantic validation**: strict schemas for all data

---

## Environmental Impact Measurement

DiagnoSys integrates **EcoLogits** to measure the environmental footprint of LLM calls:

| Metric | Unit | Description |
|--------|------|-------------|
| `energy_kwh` | kWh | Energy consumption |
| `gwp_kgCO2eq` | kg CO2 eq | Global Warming Potential |
| `adpe_mgSbEq` | mg Sb eq | Abiotic resource depletion |
| `pd_mj` | MJ | Primary energy demand |
| `wcf_liters` | liters | Water consumption footprint |

These metrics are aggregated daily per model and viewable in the statistics dashboard.

---

## Development Guidelines

### General Guidelines

- All code comments, documentation, and commit messages **must be written in English**
- Follow clean code principles: readability, simplicity, and maintainability

### Package Management

- **UV** must be used for Python package and dependency management
- Dependency versions should be explicit and reproducible (`uv add 'package'`)

### Git Workflow

- Each new feature or fix **must be developed in its own branch**, created from `main`
- Before requesting a merge:
  - Rebase or merge `main` into your branch to ensure it is up to date
  - Ensure the application runs corectly
- A **Pull Request (PR)** is required for every merge into `main`
- 2 validations by pairs required
- **No direct commits to `main`** are allowed
- Use **Issues** to monitor bugs and implement a TODO list for the team

### Code Review Policy

- **At least one code review is mandatory** before merging any PR into `main`
- Reviewers must check:
  - Code quality and readability
  - Architecture and separation of concerns
  - Typing, documentation

### Code Quality Standards

- All Python functions and methods must:
  - Be fully **type-annotated**
  - Include clear **docstrings** explaining purpose, parameters, and return values
- Use **Object-Oriented Programming (OOP)** principles:
  - Proper encapsulation
  - Clear responsibilities per class
  - No unnecessary global state

### Architecture Rules

- **Strict separation of concerns is required**:
  - Core/business logic must live in the **services or domain layer**
  - UI, routing, or API layers must remain thin
- The UI/routing layer:
  - Must not contain calculations or business rules
  - Must not implement complex logic or data transformations
  - Should only orchestrate calls to services and format input/output

> *If a piece of code does more than one thing, it probably belongs somewhere else.*

---

## Authors

Project developed as part of the SISE Master's program - University of Lyon 2

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





