import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()


class NoColorFormatter(logging.Formatter):
    """
    Custom formatter that removes ANSI color codes from log messages.
    Werkzeug adds color codes for console output which appear as garbage
    characters in log files.
    """

    # ANSI escape sequence pattern
    ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

    def format(self, record):
        # Format the record normally
        message = super().format(record)
        # Remove ANSI escape sequences
        return self.ANSI_ESCAPE.sub("", message)


def get_logging_config():
    app_name = os.getenv("APP_NAME", "diagnosys")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    return {
        "version": 1,
        "disable_existing_loggers": False,
        # Formatters define the layout of log messages
        "formatters": {
            "detailed": {
                "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s"
            },
            "no_color": {
                "()": NoColorFormatter,
                "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
            },
        },
        # handlers define where log messages go
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "detailed",
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"logs/{app_name}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "level": log_level,
                "formatter": "detailed",
            },
            "werkzeug_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/werkzeug.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
                "level": "INFO",
                "formatter": "no_color",  # Use no_color formatter for files
            },
        },
        # Define loggers for different parts of the application. getLogger(__name__) will find the correct module logger (hierarchical search).
        "loggers": {
            "werkzeug": {
                "level": "INFO",
                "handlers": ["console", "werkzeug_file"],
                "propagate": False,
            },
            "app": {
                "level": log_level,
                "handlers": ["console", "app_file"],
                "propagate": False,
            },
        },
        # Default root logger configuration if no other logger matches
        "root": {
            "level": log_level,
            "handlers": ["console", "app_file"],
        },
    }
