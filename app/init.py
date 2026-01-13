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
from app.dummy.db import DummyDB





class AppContext(Flask):
    """Typing
    Define the class of app context objects
    Cleanest way I now to enable typing on Flask app context
    """

    dummy: DummyDB


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

    # Instantiate global classes in app context
    with app.app_context():
        app.dummy = DummyDB()

    # Init pages routes
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # Init ajax endpoints + websocket
    from .ajax import ajax as ajax_blueprint, sock
    sock.init_app(app)
    app.register_blueprint(ajax_blueprint, url_prefix='/ajax')

    return app


# Application instance for production WSGI servers
# Usage: gunicorn "app:app" or waitress-serve app:app
# If imported as a module, the app instance will be used by the server.
app = create_app()
