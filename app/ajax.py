"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, current_app
from flask_sock import Sock, ConnectionClosed

from .init import AppContext


# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
ajax = Blueprint("ajax", __name__)
# Create websocket
sock = Sock()


@sock.route("/audio_stt")
def audio_stt(ws):
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
