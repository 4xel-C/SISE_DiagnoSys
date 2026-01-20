"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, render_template, jsonify, abort, current_app, request
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
    print("starting websocket")
    with open("received_audio.webm", "wb") as f:  # For testing only
        try:
            while True:
                print("received")
                data = ws.receive()
                if data is None:
                    break
                f.write(data)
        except ConnectionClosed:
            print("Audio stream ended")


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


@ajax.route("render_patient/<patient_id>", methods=["GET"])
def render_patient(patient_id: str) -> str:
    return render_template("patient.html", patient_id=patient_id)


# ---------------
# RAG


@ajax.route("process_rag", methods=["POST"])
def process_rag():
    form = request.form
    patient_id = form.get('patientId', '')
    try:
        patient_id = int(patient_id)
    except ValueError:
        abort(420, f"Bad patient_id argument {patient_id}")

    try:
        rag_result = app.rag_service.compute_rag_diagnosys(patient_id)
    except ValueError as e:
        abort(404, e)
    
    document_htmls: list[str] = []
    for document_id in rag_result.get('document_ids'):
        document = app.document_service.get_by_id(document_id)
        document_htmls.append(document.render())

    case_htmls: list[str] = []
    for patient_id in rag_result.get('related_patients_ids'):
        patient = app.patient_service.get_by_id(patient_id)
        case_htmls.append(patient.render())

    return jsonify({
        'diagnostics': rag_result.get('diagnosys'),
        'documents': document_htmls,
        'cases': case_htmls
    })



# ---------------
# DATABASE


@ajax.route("get_context/<patient_id>", methods=["GET"])
def get_context(patient_id: str):
    patient = app.patient_service.get_by_id(patient_id)

    return jsonify({"context": patient.contexte})


@ajax.route("get_results/<patient_id>", methods=["GET"])
def get_results(patient_id: str):
    patient = app.patient_service.get_by_id(patient_id)

    case_htmls: list[str] = []
    # for patient_id in patient.cases:
    #     patient = app.patient_service.get_by_id(patient_id)
    #     case_htmls.append(patient.render())

    return jsonify({
        'diagnostics': patient.diagnostic,
        'cases': case_htmls
    })


@ajax.route("update_context/<patient_id>", methods=["POST"])
def update_context(patient_id: str):
    data = request.get_json()
    context = data.get('context')
    print('CONTEXT:', context, flush=True)
    app.patient_service.update_context(patient_id, context)
    return "", 200
