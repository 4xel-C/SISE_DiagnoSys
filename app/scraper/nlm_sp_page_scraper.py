import requests
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from app.models.scraped_document import ScrapedDocument

class NLM_SP_PageScraper(BaseScraper):
    """
    Scraper for NLM StatPearls article pages.
    """
    def fetch(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse(self, raw_data: str) -> ScrapedDocument:
        soup = BeautifulSoup(raw_data, "html.parser")
        meta = soup.find("div", class_="meta-content")
        # Title
        title = ""
        if meta:
            title_span = meta.find("span", class_="title")
            if title_span:
                title = title_span.get_text(strip=True)
        # Date
        date = ""
        if meta:
            date_span = meta.find("span", itemprop="dateModified")
            if date_span:
                date = date_span.get_text(strip=True)
        # Authors
        authors = []
        if meta:
            for span in meta.find_all("span", itemprop="author"):
                authors.append(span.get_text(strip=True))
        # Affiliations
        affiliations = []
        aff_divs = soup.find_all("div", class_="affiliation")
        for aff in aff_divs:
            affiliations.append(aff.get_text(strip=True))

        # Sections
        sections = []
        for div in soup.find_all("div", id=lambda x: x and x.startswith("article-") and ".s" in x):
            # Section title
            h2 = div.find("h2")
            section_title = h2.get_text(strip=True) if h2 else ""
            # Section text
            paragraphs = [p.get_text(strip=True) for p in div.find_all("p")]
            # Objectives (li)
            objectives = [li.get_text(strip=True) for li in div.find_all("li")]
            sections.append({
                "section_title": section_title,
                "paragraphs": paragraphs,
                "objectives": objectives
            })

        # References
        references = []
        dl = soup.find("dl", class_="temp-labeled-list")
        if dl:
            for dd in dl.find_all("dd"):
                ref = {}
                bk_ref = dd.find("div", class_="bk_ref")
                if bk_ref:
                    # Full text
                    ref["full_text"] = bk_ref.get_text(" ", strip=True)
                    # Journal, year, volume, etc.
                    journal = bk_ref.find("span", class_="ref-journal")
                    ref["journal"] = journal.get_text(strip=True) if journal else ""
                    # Links
                    links = []
                    for a in bk_ref.find_all("a", href=True):
                        links.append({"url": a["href"], "label": a.get_text(strip=True)})
                    ref["links"] = links
                references.append(ref)

        # Construct ScrappedDocument
        return ScrapedDocument(
            title=title,
            date=date,
            content={
                "authors": authors,
                "affiliations": affiliations,
                "sections": sections,
                "references": references
            }
        )
    
    def validate(self, data: ScrapedDocument) -> bool:
        return super().validate(data)

    def save(self, data: ScrapedDocument, filepath: str) -> None:
        super().save(data, filepath)
