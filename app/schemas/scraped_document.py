from dataclasses import dataclass


@dataclass
class ScrapedDocument:
    """
    ScrapedDocument is the universal data model for storing scraped document content.
    """

    title: str
    date: str
    content: dict[str, str]
    link: str
