import sys
import importlib
import os
import logging.config
from app.config.logging_config import get_logging_config

logging.config.dictConfig(get_logging_config())
logger = logging.getLogger("app.pipelines")

def list_pipelines():
    pipelines_dir = os.path.dirname(__file__)
    return [
        name for name in os.listdir(pipelines_dir)
        if os.path.isdir(os.path.join(pipelines_dir, name)) and not name.startswith("__")
    ]

def run_pipeline(pipeline_name):
    try:
        # Importe le module main.py de la pipeline choisie
        pipeline_module = importlib.import_module(f"app.pipelines.{pipeline_name}.main")
        pipeline_module.run()
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de la pipeline '{pipeline_name}': {e}")

if __name__ == "__main__":
    pipelines = list_pipelines()
    logger.info(f"Pipelines disponibles: {pipelines}")
    if len(sys.argv) > 1:
        chosen = sys.argv[1]
    else:
        chosen = input("Quelle pipeline exécuter ? ")
    if chosen in pipelines:
        run_pipeline(chosen)
    else:
        logger.error("Pipeline non trouvée.")
