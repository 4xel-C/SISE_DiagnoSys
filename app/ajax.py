"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, abort, current_app, jsonify, render_template, request
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
    total: str = ""

    if patient_id is None:
        ws.close(code=1008, reason="Missing patient_id")
        return

    try:
        while True:
            # receive audio chunk
            data: bytes = ws.receive()

            # transcribe chunk
            answer = app.rag_service.transcribe_stream(data)

            # if final, send full text, else send partial
            if answer["final"]:
                ws.send(answer["text"])
                total += " " + answer["text"]
            else:
                ws.send(answer["partial"])
            print(answer)

    except ConnectionClosed:
        pass
        # Generate new context from transcribed text
        # context = app.rag_service.update_context_after_audio(patient_id, total)
        # app.patient_service.update_context(patient_id, context)


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
    for document_id in rag_result.get("document_ids", []):
        document = app.document_service.get_by_id(document_id)
        document_htmls.append(document.render())

    case_htmls: list[str] = []
    for patient_id in rag_result.get("related_patients_ids", []):
        patient = app.patient_service.get_by_id(patient_id)
        case_htmls.append(patient.render(style="case", score=0))

    return jsonify(
        {
            "diagnostics": rag_result.get("diagnosys"),
            "documents": document_htmls,
            "cases": case_htmls,
        }
    )


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
    # TEMP: fake related patient
    patient = app.patient_service.get_by_id(2)
    case_htmls.append(patient.render(style="case", score=55))
    # for related_p in patient.patients_proches:
    #     patient = app.patient_service.get_by_id(related_p.patient_id)
    #     case_htmls.append(patient.render(style='case', score=related_p.similarity_score))

    document_htmls: list[str] = []
    # TEMP: fake related patient
    document = app.document_service.get_by_id(2)
    document_htmls.append(document.render(score=71))
    # for related_d in patient.documents_proches:
    #     document = app.document_service.get_by_id(related_d.document_id)
    #     document_htmls.append(document.render(score=related_d.similarity_score))

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
