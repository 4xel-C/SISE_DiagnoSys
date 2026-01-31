"""
App routes

Only design here routes (pages) for the front end
"""

from flask import Blueprint, render_template, request, make_response
import uuid


# Create blueprint
main = Blueprint("main", __name__)


@main.route("/")
def context():
    user_id = request.cookies.get('user_id')
    show_tutorial = user_id is None

    resp = make_response(
        render_template("index.html", show_tutorial=show_tutorial)
    )

    if user_id is None:
        resp.set_cookie(
            'user_id', 
            str(uuid.uuid4()), 
            max_age=365*24*60*60, 
            secure=True, 
            samesite='Lax'
        )
        
    return resp
