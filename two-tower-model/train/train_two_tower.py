import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from models.two_tower_architecture import TwoTowerModel

# 1️⃣ Load Dataset
user_data = pd.read_csv("data/user_interactions_processed.csv")
item_data = pd.read_csv("data/item_metadata_processed.csv")

# 🔍 Debugging: Pastikan nama kolom benar
print("Kolom user_data:", user_data.columns)
print("Kolom item_data:", item_data.columns)

# 🔍 Pastikan semua book_id di user_data ada di item_data
user_data = user_data[user_data["book_id"].isin(item_data["book_id"])]

# 🔍 Debugging: Pastikan book_id dalam range benar
print("Range book_id user_data:", user_data["book_id"].min(), "-", user_data["book_id"].max())
print("Range book_id item_data:", item_data["book_id"].min(), "-", item_data["book_id"].max())

# 2️⃣ Preprocess Data
user_ids = user_data["user_id"].values
item_ids = user_data["book_id"].values
# labels = user_data["rating"].values  # Bisa diganti dengan status_baca jika lebih sesuai
labels = (user_data["rating"] >= 4).astype(int).values

# 🔍 Mapping Genre
genre_map = dict(zip(item_data["book_id"], item_data["genre"]))
user_genres = np.array([genre_map.get(book_id, 0) for book_id in item_ids])

# 3️⃣ Train-Test Split
train_user, test_user, train_item, test_item, train_labels, test_labels, train_genre, test_genre = train_test_split(
    user_ids, item_ids, labels, user_genres, test_size=0.2, random_state=42
)

# 4️⃣ Model Hyperparameters
embedding_dim = 32
num_users = user_data["user_id"].nunique() + 1  # ⬅️ Tambahkan +1 agar indeks valid
num_items = item_data["book_id"].nunique() + 1  # ⬅️ Tambahkan +1 agar indeks valid
num_genres = item_data["genre"].nunique() + 1  # ⬅️ Pastikan genre valid
num_authors = item_data["author"].nunique() + 1
num_artists = item_data["artist"].nunique() + 1

# 5️⃣ Inisialisasi Model
model = TwoTowerModel(num_users, num_items, num_genres, num_authors, num_artists, embedding_dim)

# 6️⃣ Compile Model
# model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mae"])
model.build(input_shape=[
    tf.TensorShape([None]),  # user_input
    tf.TensorShape([None]),  # genre_input
    tf.TensorShape([None])   # item_input
])

# 7️⃣ Training Model
model.fit(
    x=[train_user, train_genre, train_item],  
    y=train_labels,
    validation_data=([test_user, test_genre, test_item], test_labels),  
    batch_size=128,
    epochs=10,
    verbose=1
)

# 8️⃣ Evaluasi Model
loss, accuracy = model.evaluate(
    x=[test_user, test_genre, test_item],
    y=test_labels,
    verbose=1
)

print(f"\n✅ Evaluasi Selesai - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# 🔚 Simpan bobot model (tanpa struktur model)
model.save_weights("models/two_tower.weights.h5")

