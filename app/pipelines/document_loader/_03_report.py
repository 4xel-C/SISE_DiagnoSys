import logging

logger = logging.getLogger("app.pipelines.document_loader")


def report(results: list):
    """
    Generate a report based on the results of the document loader pipeline.
    Args:
        results (list): A list of dictionaries containing the results of the pipeline.
    """
    logger.info("--- Pipeline Report ---")
    for r in results:
        logger.info(f"{r['title']}: {r['action']}")
    logger.info("-----------------------")
