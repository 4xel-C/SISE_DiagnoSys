import importlib
import logging
import os
import sys

logger = logging.getLogger("app.pipelines")


def list_pipelines() -> list:
    """
    List all available pipelines in the app/pipelines directory.
    Returns:
        list: A list of pipeline names available in the app/pipelines directory.
    """
    pipelines_dir = os.path.dirname(__file__)
    return [
        name
        for name in os.listdir(pipelines_dir)
        if os.path.isdir(os.path.join(pipelines_dir, name))
        and not name.startswith("__")
    ]


def run_pipeline(pipeline_name: str):
    """
    Run the specified pipeline by importing its main module and executing the run function.
    Args:
        pipeline_name (str): The name of the pipeline to run.
    """
    try:
        pipeline_module = importlib.import_module(f"app.pipelines.{pipeline_name}.main")
        pipeline_module.run()
    except Exception as e:
        logger.error(f"Error running pipeline '{pipeline_name}': {e}")


if __name__ == "__main__":
    pipelines = list_pipelines()
    logger.info(f"Available pipelines: {pipelines}")
    if len(sys.argv) > 1:
        chosen = sys.argv[1]
    else:
        chosen = input("Which pipeline to run? ")
    if chosen in pipelines:
        run_pipeline(chosen)
    else:
        logger.error("Pipeline not found.")
