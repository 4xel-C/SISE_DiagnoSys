"""
Main orchestrator for the document_loader pipeline.
Executes each step in order: load, parse, update, report.
"""

import logging.config
from app.config.logging_config import get_logging_config

logging.config.dictConfig(get_logging_config())
logger = logging.getLogger("app.pipelines.document_loader")
from . import _00_load_scraped_json as step0
from . import _01_parse_documents as step1
from . import _02_update_db as step2
from . import _03_report as step3


def run():
    """
    Run the document_loader pipeline.
    """
    logger.info("[document_loader] Step 0: Loading scraped JSON files...")
    docs = step0.load_scraped_json()
    logger.info(f"Loaded {len(docs)} documents.")

    logger.info("[document_loader] Step 1: Parsing documents...")
    parsed_docs = step1.parse_documents(docs)
    logger.info(f"Parsed {len(parsed_docs)} documents.")

    logger.info("[document_loader] Step 2: Updating database...")
    update_results = step2.update_db(parsed_docs)
    logger.info(f"Database update results: {update_results}")

    logger.info("[document_loader] Step 3: Reporting...")
    step3.report(update_results)
    logger.info("Pipeline completed.")
