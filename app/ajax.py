"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

import logging
from time import time
from typing import cast

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from .init import AppContext
from .services.rag_service import UnsafeRequestException

logger = logging.getLogger(__name__)

# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
ajax = Blueprint("ajax", __name__)


# ----------------
# AUDIO TRANSCRIPTION


@ajax.route("audio_stt/<int:patient_id>", methods=["POST"])
def audio_stt(patient_id: int):
    """
    Transcribe complete audio and update patient context.
    Expects audio/webm binary data in request body.
    """
    # TIME_ID
    temp_start = time()
    # Get audio data from request body
    audio_data = request.get_data()

    if not audio_data:
        return jsonify({"error": "Aucun signal audio reçu"}), 400

    # Transcribe audio
    transcription = app.rag_service.transcribe(audio_data)

    # TIME_ID
    print(f"audio_stt - Transcription empty, took {time() - temp_start:.2f}s")
    # Update context with transcription if not empty
    if transcription and len(transcription.strip()) > 0:
        try:
            app.rag_service.update_context_after_audio(patient_id, transcription)
            return jsonify({"transcription": transcription}), 200
        except UnsafeRequestException as e:
            logger.warning(
                f"Guardrail blocked transcription for patient {patient_id}: "
                f"checkpoint={e.checkpoint}, confidence={e.confidence:.3f}"
            )
            return jsonify(
                {
                    "error": "Entrée bloquée par le filtre de sécurité",
                    "transcription": transcription,
                }
            ), 400

    return jsonify({"transcription": ""}), 200


# ----------------
# SIMULATION (CHAT)


@ajax.route("process_conversation/<int:patient_id>", methods=["POST"])
def process_conversation(patient_id: int):
    """
    Simulate audio stt processing with chatbot conversation.
    Creating a fake transcription from a message and its chatbot response and update context.
    """
    print("hello from process", flush=True)
    # Get message and its response
    message: str = request.json.get("message")
    response: str = request.json.get("response")
    print("got variables", flush=True)

    simulated_transcription = message + "\n\n" + response
    print("created transcript", flush=True)
    app.rag_service.update_context_after_audio(patient_id, simulated_transcription)
    print("by from process", flush=True)

    return "", 200


# ---------------
# RENDER POPUP


@ajax.route("custom_popup", methods=["GET"])
def custom_popup():
    params = request.args.to_dict()
    return render_template("elements/custom_popup.html", **params)

@ajax.route("create_patient_popup", methods=["GET"])
def create_patient_popup():
    return render_template("elements/create_patient_popup.html")

@ajax.route("settings_popup", methods=["GET"])
def settings_popup():
    models = app.rag_service.get_llm_models()
    threshold = app.rag_service.get_guardrail_threshold()

    return render_template(
        "elements/settings_popup.html", 
        models = models['available'], 
        selected_context = models['context'],
        selected_rag = models['rag'],
        current_threshold = threshold
    )


# ---------------
# RENDER TEMPLATES


@ajax.route("search_patients", methods=["GET"])
def search_patients():
    """
    Search patient by name with a query.
    Returns all patients if no query provided
    """
    # TIME_ID
    temp_start = time()
    query = request.args.get("query")

    if query:
        patients = app.patient_service.get_by_query(query)
    else:
        patients = app.patient_service.get_all()

    htmls = [p.render() for p in patients]
    print(f"search_patients - took {time() - temp_start:.2f}s")
    return jsonify(htmls)

@ajax.route("render_patient/<int:patient_id>", methods=["GET"])
def render_patient(patient_id: int) -> str:
    return render_template("patient.html", patient_id=patient_id)

@ajax.route("render_profile/<int:patient_id>", methods=["GET"])
def render_profile(patient_id: int):
    patient = app.patient_service.get_by_id(patient_id)
    return patient.render(style='profile')


@ajax.route("render_page/<page_name>", methods=["GET"])
def render_page(page_name: str) -> str:
    print(f"pages/{page_name}.html")
    return render_template(f"pages/{page_name}.html")


@ajax.route("render_chat", methods=["GET"])
def render_chat() -> str:
    return render_template("chat.html")


@ajax.route("render_typing_bubbles", methods=["GET"])
def render_typing_bubbles():
    return render_template("elements/typing_bubbles.html")


# ---------------
# PLOTS


@ajax.route("stat_plots", methods=["GET"])
def stat_plots():
    # treating args
    new_args = {}
    # verification of the args
    temporal_axis = request.args.get("temporal_axis")
    model_name = request.args.get("model_name")
    if temporal_axis in ["W", "M", "Y"]:
        new_args["temporal_axis"] = temporal_axis

    new_args["model_name"] = model_name if model_name else None
    # sending the args to the plot_manager interface plot method
    plots = app.plot_manager.plot_all(**new_args)  # type: ignore
    # plots : {name_of_plot: __json_string__, ...}
    # -> dict of plots
    # sending the args to the plot_manager interface kpi method
    kpis = app.plot_manager.kpis_all(**new_args)  # type: ignore
    # kpis : {name_of_kpi: {kpi_value : __str_value__, kpi_commentary: __commentary__}, ...}
    # -> dict of kpis with their values and their commentary
    return jsonify({"plots": plots, "kpis": kpis})


# ---------------
# RAG


@ajax.route("process_rag/<int:patient_id>", methods=["POST"])
def process_rag(patient_id: int):
    try:
        # TIME_ID
        temp_start = time()
        app.rag_service.compute_rag_diagnosys(patient_id)
        print(f"process_rag - compute_rag_diagnosys took {time() - temp_start:.2f}s")
        return "", 200
    except ValueError as e:
        # Patient not found
        abort(404, e)
    except UnsafeRequestException:
        return jsonify({
            "error": "Entrée bloquée par le filtre de sécurité"
        }), 400
    
@ajax.route('update_settings', methods=['POST'])
def update_settings():
    data = request.get_json()
    try:
        data['threshold'] = float(data['threshold'])
    except ValueError:
        jsonify({
            "error": f"Mauvais type de seuil, impossible de convertir {data['threshold']} en nombre float."
        })

    app.rag_service.change_llm_models(context_model=data['context-model'], rag_model=data['rag-model'])
    app.rag_service.update_guardrail_threshold(new_threshold=data['threshold'])
    return jsonify({"success": True})


# ---------------
# AGENT


@ajax.route("load_agent/<int:patient_id>", methods=["POST"])
def load_agent(patient_id: int):
    response = ""

    chat_session = app.chat_service.get_or_create_chat(patient_id)
    history = chat_session.get_history()
    if len(history) == 0:
        response = chat_session.send_initial_greeting()

    return jsonify({"message": response, "history": history})


@ajax.route("query_agent/<int:patient_id>", methods=["POST"])
def query_agent(patient_id: int):
    message: str = request.json.get("query")
    chat_session = app.chat_service.get_or_create_chat(patient_id)
    response = chat_session.send_message(message)
    return jsonify({"message": response})


# ---------------
# DATABASE


@ajax.route("create_patient", methods=["POST"])
def create_patient():
    data: dict = request.get_json()
    required_keys = {
        "nom": str,
        "prenom": str,
        "gravite": str,
        "type_maladie": str,
        "symptomes_exprimes": str,
        "fc": int,
        "fr": int,
        "spo2": float,
        "ta_systolique": int,
        "ta_diastolique": int,
        "temperature": float
    }

    for field, converter in required_keys.items():
        if field not in data:
            return jsonify({
                "error": f"Le champ '{field}' est obligatoire"
            }), 400
        try:
            data[field] = converter(data[field])
        except (ValueError, TypeError):
            return jsonify({
                "error": f"Le champ '{field}' doit être de type {converter.__name__}"
            }), 400
    
    patient = app.patient_service.create(**data)

    return jsonify({'patient_id': patient.id})

@ajax.route("get_profile/<int:patient_id>", methods=["GET"])
def get_profile(patient_id: int):
    profile = app.patient_service.get_by_id(patient_id)
    return jsonify(profile.to_metadata())

@ajax.route("get_context/<int:patient_id>", methods=["GET"])
def get_context(patient_id: int):
    context = app.patient_service.get_context(patient_id)
    return jsonify({"context": context})


@ajax.route("get_diagnostic/<int:patient_id>", methods=["GET"])
def get_diagnostic(patient_id: int):
    diagnostic = app.patient_service.get_diagnostic(patient_id)
    return jsonify({"diagnostic": diagnostic})


@ajax.route("get_related_documents/<int:patient_id>", methods=["GET"])
def get_related_documents(patient_id: int):
    document_htmls: list[str] = []
    r_documents = app.patient_service.get_documents_proches(patient_id)
    if not r_documents:
        return jsonify({"documents": []})

    for id, score in r_documents:
        document = app.document_service.get_by_id(id)
        document_htmls.append(document.render(score=round(score * 100)))

    return jsonify({"documents": document_htmls})


@ajax.route("get_related_cases/<int:patient_id>", methods=["GET"])
def get_related_cases(patient_id: int):
    case_htmls: list[str] = []
    r_patients = app.patient_service.get_patients_proches(patient_id)

    if not r_patients:
        return jsonify({"cases": []})

    for id, score in r_patients:
        patient = app.patient_service.get_by_id(id)  # type: ignore
        case_htmls.append(patient.render(style="case", score=round(score * 100)))

    return jsonify({"cases": case_htmls})


@ajax.route("update_context/<int:patient_id>", methods=["POST"])
def update_context(patient_id: int):
    data = request.get_json()
    context = data.get("context")
    app.patient_service.update_context(patient_id, context)
    return "", 200
