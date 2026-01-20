"""
Development entry point for the Flask application.

Usage:
    python run.py

This file is only for local development. For production, use a WSGI server
like Gunicorn or Waitress with the app instance from app/__init__.py
"""

from app.init import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
