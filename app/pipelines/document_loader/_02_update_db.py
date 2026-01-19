import logging.config
from app.config.logging_config import get_logging_config

logging.config.dictConfig(get_logging_config())
logger = logging.getLogger("app.pipelines.document_loader")
from app.services.document_service import DocumentService
from app.rag.vectorizer import Vectorizer
from app.rag.vector_store import VectorStore
from app.config.vector_db import CollectionType
from app.models import Document


def update_db(parsed_docs):
    results = []
    doc_service = DocumentService()
    vectorizer = Vectorizer()
    vector_store = VectorStore(CollectionType.DOCUMENTS)

    for doc in parsed_docs:
        titre = doc.get("title")
        url = doc.get("link", "")
        # Concatène tous les paragraphes des sections sélectionnées
        contenu = "\n".join(
            p for section in doc.get("sections", []) for p in section.get("paragraphs", [])
        )
        # Ignore les documents sans lien ou sans contenu (ex: Contents)
        if not url or not contenu:
            logger.info(f"Document '{titre}' ignoré (pas de lien ou pas de contenu).")
            continue
        # Recherche le document existant
        existing = doc_service.search_by_titre(titre)
        if existing:
            # Met à jour si le contenu ou le lien a changé
            db_doc = existing[0]
            if db_doc.contenu != contenu or db_doc.url != url:
                with doc_service.db_manager.session() as session:
                    document = session.query(Document).filter_by(id=db_doc.id).first()
                    document.contenu = contenu
                    document.url = url
                    session.commit()
                action = "updated"
            else:
                action = "skipped (no change)"
        else:
            # Insert
            with doc_service.db_manager.session() as session:
                new_doc = Document(titre=titre, contenu=contenu, url=url)
                session.add(new_doc)
                session.commit()
            action = "inserted"
        logger.info(f"Document '{titre}': {action}")
        results.append({"title": titre, "action": action})

        # Embedding et ajout dans ChromaDB
        if contenu:
            chunks = vectorizer.chunk_text(contenu)
            embeddings = vectorizer.generate_embeddings(chunks)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{titre}_chunk_{i}"
                metadata = {"title": titre, "url": url, "chunk_index": i}
                vector_store.add(chunk_id, chunk, metadata)
                logger.info(f"Embedded chunk {i} for '{titre}' in ChromaDB.")

    logger.info(f"Total documents processed for DB: {len(parsed_docs)}")
    return results
