"""
Flask application factory and instance.

The create_app() function initializes the Flask application with proper
logging configuration. The app instance at module level is for production
WSGI servers (Gunicorn, Waitress, etc.).
"""

import logging
import logging.config
import os

from flask import Flask

from app.config.database import db
from app.config.logging_config import get_logging_config
from app.services import DocumentService, PatientService, RagService, PlotManager


class AppContext(Flask):
    """Typing
    Define the class of app context objects
    Cleanest way I now to enable typing on Flask app context
    """

    patient_service: PatientService
    rag_service: RagService
    document_service: DocumentService
    plot_manager: PlotManager


def create_app() -> Flask:
    """
    Create and configure the Flask application instance.

    Returns:
        Flask: Configured Flask application instance
    """

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Create data directory if needed
    os.makedirs("data", exist_ok=True)

    app = Flask(__name__)

    # Initialize database
    db.init_db()

    # Instantiate services in app context
    with app.app_context():
        app.patient_service = PatientService()  # type: ignore
        app.rag_service = RagService()  # type: ignore
        app.document_service = DocumentService()  # type: ignore
        app.plot_manager = PlotManager()  # type: ignore

    # Init pages routes
    from .routes import main as main_blueprint

    app.register_blueprint(main_blueprint)

    # Init ajax endpoints
    from .ajax import ajax as ajax_blueprint

    app.register_blueprint(ajax_blueprint, url_prefix="/ajax")

    return app


# Application instance for production WSGI servers
# Usage: gunicorn "app:app" or waitress-serve app:app
# If imported as a module, the app instance will be used by the server.
app = create_app()


# Configure logging
logging.config.dictConfig(get_logging_config())
logger = logging.getLogger(__name__)
logger.info("Starting Flask application")
