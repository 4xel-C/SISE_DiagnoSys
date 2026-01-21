"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, abort, jsonify, render_template, request, current_app
from flask_sock import ConnectionClosed, Sock

from .init import AppContext

# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
ajax = Blueprint("ajax", __name__)
# Create websocket
sock = Sock()


# ----------------
# WEB SOCKETS


@sock.route("/audio_stt")
def audio_stt(ws) -> None:
    patient_id = request.args.get("patient_id", type=int)
    total: str

    if patient_id is None:
        ws.close(code=1008, reason="Missing patient_id")
        return

    try:
        while True:
            # TODO: Call stt_service with audio chunk 
            # and send transcribed string back to JS:
            data = ws.receive()
            # transcript, total = app.stt_service.transcribe_chunk(data)
            # ws.send(transcript)
    except ConnectionClosed:
        print("Audio stream ended")
        # Generate new context from transcribed text
        context = app.rag_service.update_context_after_audio(patient_id, total)
        app.patient_service.update_context(patient_id, context)


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


# ---------------
# RAG


@ajax.route("process_rag/<int:patient_id>", methods=["POST"])
def process_rag(patient_id: int):
    try:
        # Compute RAG
        rag_result = app.rag_service.compute_rag_diagnosys(patient_id)
    except ValueError as e:
        abort(404, e)

    document_htmls: list[str] = []
    for document_id in rag_result.get("document_ids", []):
        document = app.document_service.get_by_id(document_id)
        document_htmls.append(document.render())

    case_htmls: list[str] = []
    for patient_id in rag_result.get("related_patients_ids", []):
        patient = app.patient_service.get_by_id(patient_id)
        case_htmls.append(patient.render())

    return jsonify({
            "diagnostics": rag_result.get("diagnosys"),
            "documents": document_htmls,
            "cases": case_htmls,
    })


# ---------------
# DATABASE


@ajax.route("get_context/<int:patient_id>", methods=["GET"])
def get_context(patient_id: int):
    patient = app.patient_service.get_by_id(patient_id)
    return jsonify({"context": patient.contexte})


@ajax.route("get_results/<int:patient_id>", methods=["GET"])
def get_results(patient_id: int):
    patient = app.patient_service.get_by_id(patient_id)

    case_htmls: list = []
    document_htmls: list = []
    # for patient_id in patient.cases:
    #     patient = app.patient_service.get_by_id(patient_id)
    #     case_htmls.append(patient.render())

    return jsonify(
        {
            "diagnostics": patient.diagnostic,
            "cases": case_htmls,
            "documents": document_htmls,
        }
    )


@ajax.route("update_context/<int:patient_id>", methods=["POST"])
def update_context(patient_id: int):
    data = request.get_json()
    context = data.get("context")
    app.patient_service.update_context(patient_id, context)
    return "", 200
