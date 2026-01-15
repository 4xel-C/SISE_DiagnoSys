"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, current_app, jsonify, render_template
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
# TEMPLATES


@ajax.route("render_diagnostics/<patient_id>", methods=["GET"])
def render_diagnostics(patient_id: str) -> str:
    # TODO: Get diagnostic content from BDD and pass
    # the structured data to HTML template
    return render_template("diagnostics.html", patient_id=patient_id)
