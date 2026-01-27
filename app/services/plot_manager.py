import logging

from app.config import Database, db

logger = logging.getLogger(__name__)


class PlotManager():

    def __init__(self, db_manager: Database = db) -> None:
        """Initialize the PlotManager..

        Args:
            db_manager (Database, optional):  Database manager instance. Defaults to global db.
        """
        self._db_manager = db_manager # default to global db

    def hello(self) -> None:
        """Simple method to test the PlotManager.

        Returns:
            None
        """
        print("Hello from PlotManager!")
