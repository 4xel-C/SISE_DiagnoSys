import logging

from app.config import Database, db

logger = logging.getLogger(__name__)

# Singleton Pattern Implementation
# note : useless ?
class SingletonMeta(type):
    """
    Bullet-proof implementation of Singleton.
    Ensures a class has only one instance and provides a global point of access to it.
    code from : 
    https://refactoring.guru/fr/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(cls, *args, **kwargs): # type: ignore
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances: # type: ignore
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance # type: ignore
        return cls._instances[cls] # type: ignore

# PlotManager Class
class PlotManager(metaclass=SingletonMeta):

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
