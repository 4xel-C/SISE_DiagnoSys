"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, render_template, jsonify, current_app, request
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

@ajax.route("search_patients", methods=['GET'])
def search_patients():
    """
    Search patient by name with a query.
    Returns all patients if no query provided
    """

    query = request.args.get('query')

    if query:
        patients = app.patient_service.get_by_query(query)
    else:
        patients = app.patient_service.get_all()

    htmls = [
        render_template(
            'patient_result.html',
            id=p.id,
            name=p.prenom,
            last_name=p.nom,
            initials=p.initials
        ) for p in patients
    ]
    return jsonify(htmls)

@ajax.route("render_diagnostics/<patient_id>", methods=["GET"])
def render_diagnostics(patient_id: str) -> str:
    # TODO: Get diagnostic content from BDD and pass
    # the structured data to HTML template
    return render_template("diagnostics.html", patient_id=patient_id)



# ---------------
# RAG

@ajax.route('process_rag', methods=['POST'])
def process_rag():
    form = request.form
    patient_id = form.get('patientId')
    context = form.get('context')
    
    # TODO: process RAG and retrive `diagnostics`, `documents` and `cases`
    diagnostics: dict = None
    documents: dict = None
    cases: dict = None

    return jsonify({
        'diagnostics': diagnostics,
        'documents': documents,
        'cases': cases
    })