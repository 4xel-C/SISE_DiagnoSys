"""
App routes

Only design here routes (pages) for the front end
"""

from flask import Blueprint, render_template


# Create blueprint
main = Blueprint("main", __name__)


@main.route("/")
def context():
    return render_template("index.html")
