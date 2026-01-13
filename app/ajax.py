"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from typing import cast

from flask import Blueprint, current_app
from flask_sock import Sock

from .init import AppContext




# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
ajax = Blueprint('ajax', __name__)
# Create websocket
sock = Sock()




@sock.route('/audio_stt')
def audio_stt(ws):
    with open("received_audio.webm", "wb") as f:   # For testing only
        while True:
            data = ws.receive()
            if data is None:
                break
            f.write(data)
    print("Audio stream ended")