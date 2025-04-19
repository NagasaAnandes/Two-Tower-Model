# preprocessing/preprocess.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_item_metadata(input_path="../data/item_metadata.csv", output_path="../data/item_metadata_processed.csv"):
    df = pd.read_csv(input_path)

    # --- Normalize & fill missing ---
    df['title'] = df['title'].fillna("").str.lower().str.strip()
    df['author'] = df['author'].fillna("").str.lower().str.strip()
    df['artist'] = df['artist'].fillna("").str.lower().str.strip()
    df['status'] = df['status'].fillna("unknown").str.lower()
    df['demographic'] = df['demographic'].fillna("unknown").str.lower()
    df['content_rating'] = df['content_rating'].fillna("safe").str.lower()

    # --- Process genres ---
    # Split string genre jadi list
    df['genres'] = df['genres'].fillna("").apply(lambda x: [g.strip().lower() for g in x.split(",") if g.strip()])

    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=[f"genre_{g}" for g in mlb.classes_])

    # --- Gabungkan ke df utama ---
    df = pd.concat([df.drop(columns=['genres']), genre_df], axis=1)

    # --- Simpan hasil ---
    df.to_csv(output_path, index=False)
    print(f"Processed metadata saved to: {output_path}")

if __name__ == "__main__":
    preprocess_item_metadata()
