"""
Flask application factory and instance.

The create_app() function initializes the Flask application with proper
logging configuration. The app instance at module level is for production
WSGI servers (Gunicorn, Waitress, etc.).
"""

import logging.config
import os

from flask import Flask

from app.config.logging_config import get_logging_config


def create_app() -> Flask:
    """
    Create and configure the Flask application instance.

    Returns:
        Flask: Configured Flask application instance
    """
    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.config.dictConfig(get_logging_config())

    app = Flask(__name__)

    logger = logging.getLogger(__name__)
    logger.info("Starting Flask application")

    return app


# Application instance for production WSGI servers
# Usage: gunicorn "app:app" or waitress-serve app:app
# If imported as a module, the app instance will be used by the server.
app = create_app()
