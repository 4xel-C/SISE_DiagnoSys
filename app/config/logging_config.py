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


def get_logging_config() -> dict:
    """Return the full configuration dictionnary for the logger. Set up the whole logger hierarchy.
    Used in the application initialization to configure logging.

    Returns:
        dict: _description_
    """
    app_name = os.getenv("APP_NAME", "diagnosys")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    print_console = os.getenv("PRINT_CONSOLE_LOGS", "true").lower() == "true"

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
                "stream": "ext://sys.stdout",
            },
            "app_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"logs/{app_name}.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 2,
                "level": log_level,
                "formatter": "detailed",
                "encoding": "utf-8",  # Correction pour Unicode
            },
            "werkzeug_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/werkzeug.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 1,
                "level": "INFO",
                "formatter": "no_color",  # Use no_color formatter for files
                "encoding": "utf-8",  # Correction pour Unicode
            },
            "config_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/config.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 1,
                "level": log_level,
                "formatter": "detailed",
                "encoding": "utf-8",  # Correction pour Unicode
            },
            "service_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/service.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 1,
                "level": log_level,
                "formatter": "detailed",
                "encoding": "utf-8",  # Correction pour Unicode
            },
            "rag_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/rag.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 1,
                "level": log_level,
                "formatter": "detailed",
                "encoding": "utf-8",  # Correction pour Unicode
            },
            "scrapper_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/scraper.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 1,
                "level": log_level,
                "formatter": "detailed",
                "encoding": "utf-8",  # Correction pour Unicode
            },
        },
        # Define loggers for different parts of the application. getLogger(__name__) will find the correct module logger (hierarchical search).
        "loggers": {
            "werkzeug": {
                "level": "INFO",
                "handlers": ["console", "werkzeug_file"]
                if print_console
                else ["werkzeug_file"],
                "propagate": False,
            },
            "app": {
                "level": log_level,
                "handlers": ["console", "app_file"] if print_console else ["app_file"],
                "propagate": False,
            },
            "app.config": {
                "level": log_level,
                "handlers": ["console", "config_file"]
                if print_console
                else ["config_file"],
                "propagate": True,
            },
            "app.service": {
                "level": log_level,
                "handlers": ["console", "service_file"]
                if print_console
                else ["service_file"],
                "propagate": True,
            },
            "app.rag": {
                "level": log_level,
                "handlers": ["console", "rag_file"] if print_console else ["rag_file"],
                "propagate": True,
            },
            "app.scraper": {
                "level": log_level,
                "handlers": ["console", "scrapper_file"]
                if print_console
                else ["scrapper_file"],
                "propagate": True,
            },
        },
        # Default root logger configuration if no other logger matches
        "root": {
            "handlers": ["console", "app_file"] if print_console else ["app_file"],
        },
    }
