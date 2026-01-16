from datetime import datetime
import requests
from bs4 import BeautifulSoup
from app.models.scraped_document import ScrapedDocument
from .base_scraper import BaseScraper

class NLM_StatPearlsScraper(BaseScraper):
    """
    Scraper for NLM StatPearls articles.
    """
    def fetch(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse(self, raw_data: str) -> ScrapedDocument:
        soup = BeautifulSoup(raw_data, 'html.parser')
        # Find the contents section header
        contents_h2 = soup.find("h2", string="Contents")
        if not contents_h2:
            raise ValueError("Contents section not found in the document.")
        # Search ul after the h2
        ul = contents_h2.find_next("ul")
        if not ul:
            raise ValueError("Contents list not found in the document.")
        
        items = []
        for li in ul.find_all("li"):
            a = li.find("a")
            if a and a.get("href"):
                items.append({
                    "text": a.get_text(strip=True),
                    "url": a["href"]
                })
            else:
                items.append({
                    "text": li.get_text(strip=True),
                    "url": None
                })
        return ScrapedDocument(
            title="Contents",
            date=datetime.now().strftime("%Y-%m-%d"),
            content={"items": items}
        )
    
    def validate(self, data: ScrapedDocument) -> bool:
        return super().validate(data)

    def save(self, data: ScrapedDocument, filepath: str) -> None:
        super().save(data, filepath)
