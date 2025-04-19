import requests
import pandas as pd
import urllib3
import time
from tqdm import tqdm

# Supress SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_manga_data(limit=100, offset=0, verify_ssl=False, max_retries=3, retry_delay=5):
    base_url = "https://api.mangadex.org/manga"
    params = {
        "limit": limit,
        "offset": offset,
        "includes[]": ["author", "artist"],
        "contentRating[]": ["safe", "suggestive", "erotica"],
        "order[createdAt]": "desc",
        "availableTranslatedLanguage[]": "en"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, verify=verify_ssl, timeout=10)
            response.raise_for_status()
            data = response.json()["data"]

            records = []
            for manga in data:
                attributes = manga["attributes"]
                relationships = manga["relationships"]

                title = attributes["title"].get("en") or list(attributes["title"].values())[0]
                author = next((rel["attributes"]["name"] for rel in relationships if rel["type"] == "author" and "attributes" in rel), None)
                artist = next((rel["attributes"]["name"] for rel in relationships if rel["type"] == "artist" and "attributes" in rel), None)

                genres = [tag["attributes"]["name"]["en"] for tag in attributes.get("tags", []) if tag["attributes"]["group"] == "genre"]
                demographic = attributes.get("publicationDemographic")
                status = attributes.get("status")
                rating = attributes.get("contentRating")

                records.append({
                    "title": title,
                    "author": author,
                    "artist": artist,
                    "genres": genres,
                    "status": status,
                    "demographic": demographic,
                    "content_rating": rating
                })

            return pd.DataFrame(records)

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise

def fetch_all_manga_data(total=1000, batch_size=100, save_path="../data/item_metadata.csv"):
    all_data = pd.DataFrame()
    for offset in tqdm(range(0, total, batch_size)):
        df = fetch_manga_data(limit=batch_size, offset=offset, verify_ssl=False)
        all_data = pd.concat([all_data, df], ignore_index=True)
        time.sleep(1.2)  # biar ga kena rate-limit
    all_data.to_csv(save_path, index=False)
    print(f"✅ Saved {len(all_data)} manga entries to {save_path}")
    return all_data

if __name__ == "__main__":
    fetch_all_manga_data(total=2000, batch_size=100, save_path="../data/item_metadata.csv")
