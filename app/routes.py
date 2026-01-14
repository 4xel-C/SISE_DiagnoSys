"""
App routes

Only design here routes (pages) for the front end
"""

from flask import Blueprint, render_template, current_app
from typing import cast

from .init import AppContext




# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
main = Blueprint('main', __name__)




@main.route('/')
def context():
    patients = app.dummy.patients()
    return render_template('index.html', patients=patients)