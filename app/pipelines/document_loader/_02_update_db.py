import logging

from app.services.document_service import DocumentService

logger = logging.getLogger("app.pipelines.document_loader")


def update_db(parsed_docs: list) -> list:
    """
    Update the database with parsed documents.
    Args:
        parsed_docs (list): A list of parsed document dictionaries.

    Returns:
        list: A list of results indicating the action taken for each document.
    """
    results = []
    doc_service = DocumentService()

    for doc in parsed_docs:
        titre = doc.get("title")
        url = doc.get("link", "")

        contenu = "\n".join(
            p
            for section in doc.get("sections", [])
            for p in section.get("paragraphs", [])
        )

        # Ignore documents without URL or content
        if not url or not contenu:
            logger.info(f"Document '{titre}' skipped due to missing URL or content.")
            continue

        existing = doc_service.search_by_titre(titre)
        if existing:
            db_doc = existing[0]
            # Update if content or URL has changed
            if db_doc.contenu != contenu or db_doc.url != url:
                doc_service.update_document(
                    document_id=db_doc.id, titre=db_doc.titre, contenu=contenu, url=url
                )
                action = "updated"
            else:
                action = "skipped (no change)"
        else:
            # Insert
            doc_service.create(titre=titre, contenu=contenu, url=url)
            action = "inserted"
        logger.info(f"Document '{titre}': {action}")
        results.append({"title": titre, "action": action})

    logger.info(f"Total documents processed for DB: {len(parsed_docs)}")
    return results
