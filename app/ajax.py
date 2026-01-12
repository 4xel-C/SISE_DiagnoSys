"""
AJAX endpoints

Only design here function designed to be called from
front end. No complex logic.
"""

from flask import Blueprint, current_app
from typing import cast

from .init import AppContext




# Cast app_context typing
app = cast(AppContext, current_app)
# Create blueprint
ajax = Blueprint('ajax', __name__)




