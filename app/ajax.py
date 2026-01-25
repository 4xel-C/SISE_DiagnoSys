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

    # validate patient_id
    if patient_id is None:
        ws.close(code=1008, reason="Missing patient_id")
        return

    model = getattr(app.rag_service, "asr_model", None)
    try:
        # start per-websocket session if supported
        if model and hasattr(model, "start_session"):
            try:
                model.start_session()
            except Exception:
                pass

        while True:
            # receive audio chunk
            data = ws.receive()
            if data is None:
                break

            # transcribe chunk
            answer = app.rag_service.transcribe_stream(data)

            # if final, send full text, else send partial
            ws.send(answer["text"])
            if answer["final"]:
                total += " " + answer["text"]
            print("ASR answer: %s", answer)

    except ConnectionClosed:
        pass
    finally:
        # End session and get final result
        if model and hasattr(model, "end_session"):
            try:
                final = model.end_session()
                if final and final.get("text"):
                    total += " " + final.get("text")
                    try:
                        ws.send(final.get("text"))
                    except Exception:
                        pass
            except Exception:
                pass
        # Update context with complete transcription
        if len(total.strip()) > 0:
            context = app.rag_service.update_context_after_audio(patient_id, total)
            app.patient_service.update_context(patient_id, context)
        else:
            pass



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
    context = app.patient_service.get_context(patient_id)
    return jsonify({
        "context": context
    })

@ajax.route("get_diagnostic/<int:patient_id>", methods=["GET"])
def get_diagnostic(patient_id: int):
    diagnostic = app.patient_service.get_diagnostic(patient_id)
    return jsonify({
        'diagnostic': diagnostic
    })

@ajax.route("get_related_documents/<int:patient_id>", methods=["GET"])
def get_related_documents(patient_id: int):
    document_htmls: list[str] = []
    r_documents = app.patient_service.get_documents_proches(patient_id)
    for id, score in r_documents:
        document = app.document_service.get_by_id(id) #type: ignore
        document_htmls.append(
            document.render(
                score=round(score * 100)
            )
        )

    return jsonify({
        "documents": document_htmls
    })

@ajax.route("get_related_cases/<int:patient_id>", methods=["GET"])
def get_related_cases(patient_id: int):
    case_htmls: list[str] = []
    r_patients = app.patient_service.get_patients_proches(patient_id)
    for id, score in r_patients:
        patient = app.patient_service.get_by_id(id) #type: ignore
        case_htmls.append(
            patient.render(
                style='case', 
                score=round(score * 100)
            )
        )
        
    return jsonify({
        "cases": case_htmls
    })

@ajax.route("update_context/<int:patient_id>", methods=["POST"])
def update_context(patient_id: int):
    print('updating', patient_id)
    data = request.get_json()
    context = data.get("context")
    print('context', context)
    app.patient_service.update_context(patient_id, context)
    return "", 200
