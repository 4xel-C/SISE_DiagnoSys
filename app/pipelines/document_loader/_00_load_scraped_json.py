import os
import json
import logging.config
from app.config.logging_config import get_logging_config

logging.config.dictConfig(get_logging_config())
logger = logging.getLogger("app.pipelines.document_loader")

def load_scraped_json(folder_path="data/scraped_documents"):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    documents.append(doc)
                logger.info(f"Loaded file: {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Total loaded documents: {len(documents)}")
    return documents
