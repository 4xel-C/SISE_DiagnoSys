from .base_scraper import BaseScraper
from app.models.scraped_document import ScrapedDocument

class DummyScraper(BaseScraper):
    """
    A dummy scraper implementation for testing purposes.
    """
    def fetch(self, url: str) -> str:
        # Return dummy raw data
        return "Titre: Exemple\nDate: 2026-01-15\nContenu: Ceci est un test."

    def parse(self, raw_data: str) -> ScrapedDocument:
        # Simple parsing logic for the dummy data
        return ScrapedDocument(title="Exemple", date="2026-01-15", content={"text": "Ceci est un test."})

    def validate(self, data: ScrapedDocument) -> bool:
        return super().validate(data)

    def save(self, data: ScrapedDocument, filepath: str) -> None:
        super().save(data, filepath)

if __name__ == "__main__":
    scraper = DummyScraper()
    scraper.scrape("http://dummy.url")
    print("Scraping completed.")
