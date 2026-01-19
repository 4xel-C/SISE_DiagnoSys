from app.scraper.nlm_sp_scraper import NLM_StatPearlsScraper
from app.scraper.nlm_sp_page_scraper import NLM_SP_PageScraper
from datetime import datetime
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    sp_scrapper = NLM_StatPearlsScraper()
    sp_scrapper.scrape("https://www.ncbi.nlm.nih.gov/books/NBK430685/")

    json_path = os.path.join(
        "data",
        "scraped_documents",
        f"Contents_{sp_scrapper.parse.__annotations__.get('date', datetime.now().strftime('%Y-%m-%d'))}.json",
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_links = []
    if "items" in data.get("content", {}):
        for item in data["content"]["items"]:
            if item["url"]:
                url = item["url"]
                if url.startswith("/"):
                    url = "https://www.ncbi.nlm.nih.gov" + url
                document_links.append(url)

    print(f"Found {len(document_links)} document links to scrape.")

    # page_scraper = NLM_SP_PageScraper()

    # def scrape_link(link):
    #     try:
    #         page_scraper.scrape(link)
    #         print(f"Scraped: {link}")
    #     except Exception as e:
    #         print(f"Erreur lors du scraping de {link}: {e}")
    #     time.sleep(0.5)

    # with ThreadPoolExecutor(max_workers=8) as executor:
    #     futures = [executor.submit(scrape_link, link) for link in document_links]
    #     total = len(futures)
    #     completed = 0
    #     for future in as_completed(futures):
    #         completed += 1
    #         print(f"Remaining tasks: {total - completed}")

    # --- Ajout du lien unique dans chaque JSON de document ---
    import glob
    scraped_folder = os.path.join("data", "scraped_documents")
    # On crée un mapping titre -> url à partir du fichier Contents
    title_to_url = {}
    for item in data.get("content", {}).get("items", []):
        if item.get("text") and item.get("url"):
            # On nettoie le titre pour matcher (strip, lower)
            title_to_url[item["text"].strip().lower()] = item["url"]

    for json_file in glob.glob(os.path.join(scraped_folder, "*.json")):
        base = os.path.basename(json_file)
        if base.startswith("Contents_"):
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        doc_title = doc_data.get("title", "").strip().lower()
        matched_url = title_to_url.get(doc_title)
        # Si trouvé, on ajoute le lien complet
        if matched_url:
            if matched_url.startswith("/"):
                matched_url = "https://www.ncbi.nlm.nih.gov" + matched_url
        # Si pas trouvé, None
        doc_data["link"] = matched_url
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=4)
        print(f"Lien ajouté dans {json_file} : {matched_url}")


if __name__ == "__main__":
    print("Starting scraping process...")
    main()
    print("Scraping process completed.")

    # Zip all scraped documents
    # import zipfile

    # def zip_scraped_documents(zip_name="scraped_documents.zip"):
    #     folder = os.path.join("data", "scraped_documents")
    #     with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    #         for filename in os.listdir(folder):
    #             if filename.endswith(".json"):
    #                 zipf.write(os.path.join(folder, filename), arcname=filename)
    #     print(f"Archive créée : {zip_name}")

    # zip_scraped_documents()
