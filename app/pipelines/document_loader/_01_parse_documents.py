import logging.config
from app.config.logging_config import get_logging_config

logging.config.dictConfig(get_logging_config())
logger = logging.getLogger("app.pipelines.document_loader")


def parse_documents(raw_docs: list) -> list:
    """
    Parse raw documents to extract relevant sections based on target types.
    Args:
        raw_docs (list): List of raw document dictionaries.

    Returns:
        list: List of parsed document dictionaries with relevant sections.
    """
    parsed = []
    for doc in raw_docs:
        relevant_sections = doc.get("content", {}).get("sections", [])
        link = doc.get("link")
        for section in relevant_sections:
            logger.info(
                f"Section '{section.get('section_title', '')}' found in doc '{doc.get('title', '')}'"
            )
        parsed.append(
            {
                "title": doc.get("title"),
                "date": doc.get("date"),
                "sections": relevant_sections,
                "link": link,
            }
        )
        logger.info(
            f"Parsed document: {doc.get('title', '')} with {len(relevant_sections)} sections."
        )
    logger.info(f"Total parsed documents: {len(parsed)}")
    return parsed
