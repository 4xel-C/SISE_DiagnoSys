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
    total = None

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

        if total is not None:
            context = app.rag_service.update_context_after_audio(patient_id, total)
            app.patient_service.update_context(patient_id, context)
        else:
            print("No transcription total available to update context.")



# ---------------
# RENDER POPUP

@ajax.route("custom_popup", methods=["GET"])
def custom_popup():
    params = request.args.to_dict()
    print(params)
    return render_template('elements/custom_popup.html', **params)


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
        rag_result = app.rag_service.compute_rag_diagnosys(patient_id)
    except ValueError as e:
        # Patient not found
        abort(404, e)

    document_htmls: list[str] = []
    for document_id, document_score in rag_result['related_documents']:
        document = app.document_service.get_by_id(document_id)
        document_htmls.append(document.render(score=document_score))

    case_htmls: list[str] = []
    for patient_id, patient_score in rag_result['related_patients']:
        patient = app.patient_service.get_by_id(patient_id)
        case_htmls.append(patient.render(style='case', score=patient_score))

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

    case_htmls: list[str] = []
    related_patients = app.patient_service.get_patients_proches(patient.id)
    for related_p in related_patients:
        patient = app.patient_service.get_by_id(related_p.patient_id)
        case_htmls.append(patient.render(style='case', score=related_p.formatted_score))

    document_htmls: list[str] = []
    related_documents = app.patient_service.get_documents_proches(patient.id)
    for related_d in related_documents:
        document = app.document_service.get_by_id(related_d.document_id)
        document_htmls.append(document.render(score=related_d.formatted_score))

    return jsonify({
        "diagnostics": patient.diagnostic,
        "cases": case_htmls,
        "documents": document_htmls,
    })


@ajax.route("update_context/<int:patient_id>", methods=["POST"])
def update_context(patient_id: int):
    data = request.get_json()
    context = data.get("context")
    app.patient_service.update_context(patient_id, context)
    return "", 200
