from abc import ABC, abstractmethod
from app.models.scraped_document import ScrapedDocument
import os
import json
import re

class BaseScraper(ABC):
    """
    Abstract base class for document scrapers.
    """
    @abstractmethod
    def fetch(self, url: str) -> str: # type: ignore
        """
        Fetch raw data from the given URL.

        Args:
            url (str): The URL to fetch data from.
        Returns:
            str: The fetched raw data.
        """
        pass   

    @abstractmethod
    def parse(self, raw_data: str) -> ScrapedDocument: # type: ignore
        """
        Parse raw data into JSON format.

        Args:
            raw_data (str): The raw data to parse.

        Returns:
            json: The parsed document content in JSON format.
        """
        pass 

    @abstractmethod
    def validate(self, data: ScrapedDocument) -> bool: # type: ignore
        """
        Validate the scraped document content.

        Args:
            data (ScrapedDocument): The scraped document content in ScrapedDocument format.

        Returns:
            bool: True if the content is valid, False otherwise.
        """
        if not data.title or not data.date or not data.content:
            return False

        return True

    @abstractmethod
    def save(self, data: ScrapedDocument, filepath: str) -> None: # type: ignore
        """
        Save the scraped document content to a file.

        Args:
            data (ScrapedDocument): The scraped document content in ScrapedDocument format.
            filepath (str): The file path where the content should be saved.
        """
        save_dir = os.path.join("data", "scraped_documents")
        os.makedirs(save_dir, exist_ok=True)
        # Clean filename
        safe_filename = re.sub(r'[\\/:*?"<>|,]', '_', filepath)
        filepath = os.path.join(save_dir, safe_filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data.__dict__, f, indent=4, ensure_ascii=False)

    def scrape(self, url: str) -> None:
        """
        Orchestrates the scraping process: fetch, parse, validate, and save.

        Args:
            url (str): The URL to scrape data from.
        """
        raw_data = self.fetch(url)
        parsed_data = self.parse(raw_data)
        if self.validate(parsed_data):
            self.save(parsed_data, f"{parsed_data.title}_{parsed_data.date}.json")
        else:
            raise ValueError("Invalid data scraped from the URL.")
