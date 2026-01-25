"""
Development entry point for the Flask application.

Usage:
    python run.py

This file is only for local development. For production, use a WSGI server
like Gunicorn or Waitress with the app instance from app/__init__.py
"""

import os
import sys
venv_ort = os.path.join(sys.prefix, "Lib", "site-packages", "onnxruntime", "capi")
if os.path.isdir(venv_ort):
    os.add_dll_directory(venv_ort)

from app.init import app

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000, threaded=True)
