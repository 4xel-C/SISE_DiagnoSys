"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from .init import AppContext

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
    # Get audio data from request body
    audio_data = request.get_data()

    if not audio_data:
        return jsonify({"error": "No audio data received"}), 400

    # Transcribe audio
    transcription = app.rag_service.transcribe(audio_data)

    # Update context with transcription if not empty
    if transcription and len(transcription.strip()) > 0:
        app.rag_service.update_context_after_audio(patient_id, transcription)
        return jsonify({"transcription": transcription}), 200

    return jsonify({"transcription": ""}), 200



# ---------------
# RENDER POPUP


@ajax.route("custom_popup", methods=["GET"])
def custom_popup():
    params = request.args.to_dict()
    return render_template("elements/custom_popup.html", **params)



# ---------------
# RENDER TEMPLATES


@ajax.route("search_patients", methods=["GET"])
def search_patients():
    """
    Search patient by name with a query.
    Returns all patients if no query provided
    """
    query = request.args.get("query")

    if query:
        patients = app.patient_service.get_by_query(query)
    else:
        patients = app.patient_service.get_all()

    htmls = [p.render() for p in patients]
    return jsonify(htmls)


@ajax.route("render_patient/<int:patient_id>", methods=["GET"])
def render_patient(patient_id: int) -> str:
    return render_template("patient.html", patient_id=patient_id)


@ajax.route("render_chat", methods=["GET"])
def render_chat() -> str:
    return render_template("chat.html")

@ajax.route("render_typing_bubbles", methods=["GET"])
def render_typing_bubbles():
    return render_template("elements/typing_bubbles.html")



# ---------------
# RAG


@ajax.route("process_rag/<int:patient_id>", methods=["POST"])
def process_rag(patient_id: int):
    try:
        app.rag_service.compute_rag_diagnosys(patient_id)
        return "", 200
    except ValueError as e:
        # Patient not found
        abort(404, e)



# ---------------
# AGENT


@ajax.route("load_agent/<int:patient_id>", methods=["POST"])
def load_agent(patient_id: int):
    response = ""
    
    chat_session = app.chat_service.get_or_create_chat(patient_id)
    history = chat_session.get_history()
    if len(history) == 0:
        response = chat_session.send_initial_greeting()

    return jsonify({
        'message': response,
        'history': history
        })

@ajax.route("query_agent/<int:patient_id>", methods=["POST"])
def query_agent(patient_id: int):
    message: str = request.json.get('query')
    chat_session = app.chat_service.get_or_create_chat(patient_id)
    response = chat_session.send_message(message)
    return jsonify({'message': response})



# ---------------
# DATABASE

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
    print("updating", patient_id)
    data = request.get_json()
    context = data.get("context")
    print("context", context)
    app.patient_service.update_context(patient_id, context)
    return "", 200
