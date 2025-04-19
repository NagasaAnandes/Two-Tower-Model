import pandas as pd
import numpy as np
import tensorflow as tf
import os
from models.two_tower_architecture import TwoTowerModel

# Load data
user_data = pd.read_csv("data/user_interactions_processed.csv")
item_data = pd.read_csv("data/item_metadata_processed.csv")

# Hitung parameter model
num_users = user_data["user_id"].nunique() + 1
num_items = item_data["book_id"].nunique() + 1
num_genres = item_data["genre"].nunique() + 1
embedding_dim = 32

# Path model
FULL_MODEL_PATH = "models/two_tower_trained.h5"
WEIGHTS_PATH = "models/two_tower.weights.h5"

# Load model (full atau weights)
if os.path.exists(FULL_MODEL_PATH):
    print("✅ Load dari FULL MODEL...")
    model = tf.keras.models.load_model(
        FULL_MODEL_PATH,
        custom_objects={"TwoTowerModel": TwoTowerModel}
    )
else:
    print("✅ Load dari WEIGHTS saja...")
    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        num_authors=1,  # tidak dipakai
        num_artists=1,  # tidak dipakai
        embedding_dim=embedding_dim
    )
    # Build dulu sebelum load weights
    model.build(input_shape=[
        tf.TensorShape([None]),  # user_input
        tf.TensorShape([None]),  # genre_input
        tf.TensorShape([None])   # item_input
    ])

    # Load weights
    model.load_weights(WEIGHTS_PATH)

# Fungsi rekomendasi
def get_recommendations(user_id, top_n=5):
    user_history = user_data[user_data["user_id"] == user_id]
    if not user_history.empty:
        top_rated = user_history[user_history["rating"] == user_history["rating"].max()]
        mapped_genres = top_rated["book_id"].map(
            dict(zip(item_data["book_id"], item_data["genre"]))
        )
        if not mapped_genres.dropna().empty:
            favorite_genre = mapped_genres.mode()[0]
        else:
            favorite_genre = 0  # fallback genre default
    else:
        favorite_genre = 0

    book_ids = item_data["book_id"].values
    user_ids = np.array([user_id] * len(book_ids))
    user_genres = np.array([favorite_genre] * len(book_ids))

    predictions = model.predict([user_ids, user_genres, book_ids], verbose=0)
    top_indices = predictions.flatten().argsort()[::-1][:top_n]
    top_books = item_data.iloc[top_indices]
    top_rated = user_history[user_history["rating"] == user_history["rating"].max()]


    return top_books[["book_id", "genre", "author", "artist"]]

# Contoh penggunaan
if __name__ == "__main__":
    user_id = 1
    print(f"📚 Top Rekomendasi untuk User ID {user_id}")
    print(get_recommendations(user_id))
