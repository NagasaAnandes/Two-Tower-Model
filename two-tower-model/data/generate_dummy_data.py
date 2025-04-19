import pandas as pd
import numpy as np
import random

# Dummy User Interactions
user_data = {
    "user_id": np.arange(1, 21),  # 20 User
    "book_id": np.random.randint(100, 500, 20),
    "rating": np.round(np.random.uniform(1, 5, 20), 1),
    "status_baca": np.random.choice(["Ingin dibaca", "Sedang dibaca", "Belum dibaca"], 20),
    "timestamp": np.random.randint(1700000000, 1720000000, 20)
}
df_user = pd.DataFrame(user_data)

# Dummy Item Metadata
genres = ["Action", "Comedy", "Drama", "Fantasy", "Horror"]
authors = ["John Smith", "Alice Kim", "Tom White", "Emma Brown", "David Clark"]
artists = ["Jane Doe", "Bob Brown", "Anna Green", "Mike Lee", "Sarah Adams"]

item_data = {
    "book_id": np.arange(100, 120),  # 20 Komik
    "genre": [random.choice(genres) for _ in range(20)],
    "author": [random.choice(authors) for _ in range(20)],
    "artist": [random.choice(artists) for _ in range(20)],
    "year": np.random.randint(2015, 2024, 20),
    "trending": np.random.choice([0, 1], 20),
    "new_release": np.random.choice([0, 1], 20),
    "top_rated": np.random.choice([0, 1], 20)
}
df_item = pd.DataFrame(item_data)

# Simpan ke CSV
df_user.to_csv("data/user_interactions.csv", index=False)
df_item.to_csv("data/item_metadata.csv", index=False)

print("✅ Dummy dataset berhasil dibuat!")
