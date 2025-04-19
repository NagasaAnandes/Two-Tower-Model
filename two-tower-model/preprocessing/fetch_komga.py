import requests
import pandas as pd

# =========================
# 🔧 CONFIG
# =========================
BASE_URL = "http://localhost:25600"
EMAIL = "admin@komga.org"  # ganti sesuai user Komga kamu
PASSWORD = "komga"         # ganti sesuai password

# =========================
# 🔐 Login ke Komga
# =========================
def login():
    login_url = f"{BASE_URL}/api/v1/auth/login"
    session = requests.Session()
    payload = {"email": EMAIL, "password": PASSWORD}
    resp = session.post(login_url, json=payload)

    if resp.status_code == 200:
        print("✅ Login sukses.")
        return session
    else:
        print("❌ Login gagal:", resp.text)
        exit()

# =========================
# 📦 Fetch Series Data
# =========================
def fetch_series(session, limit=100):
    all_records = []
    page = 0

    while len(all_records) < limit:
        url = f"{BASE_URL}/api/v1/series"
        params = {"page": page, "size": 50}
        resp = session.get(url, params=params)
        resp.raise_for_status()

        data = resp.json()["content"]
        if not data:
            break  # tidak ada lagi data

        for series in data:
            meta = series.get("metadata", {})
            title = series.get("title", "")
            tags = meta.get("tags", [])
            status = meta.get("status")
            age_rating = meta.get("ageRating")
            authors = meta.get("authors", [])

            # ambil author dan artist sesuai role
            author_names = [a["name"] for a in authors if a["role"] == "writer"]
            artist_names = [a["name"] for a in authors if a["role"] in ["penciller", "inker", "artist"]]

            all_records.append({
                "title": title,
                "genres": tags,
                "status": status,
                "age_rating": age_rating,
                "authors": author_names,
                "artists": artist_names
            })

            if len(all_records) >= limit:
                break

        page += 1

    return pd.DataFrame(all_records)

# =========================
# 🏁 Main
# =========================
if __name__ == "__main__":
    session = login()
    df = fetch_series(session, limit=200)
    df.to_csv("data/komga_metadata.csv", index=False)
    print("✅ Data berhasil disimpan ke data/komga_metadata.csv")
