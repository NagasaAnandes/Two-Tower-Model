import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model

class UserTower(Model):
    def __init__(self, num_users, num_genres, embedding_dim=32):
        super(UserTower, self).__init__()

        # Embedding untuk user_id
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name="user_embedding")

        # Embedding untuk genre preferensi
        self.genre_embedding = Embedding(input_dim=num_genres, output_dim=embedding_dim, name="genre_embedding")

        # Dense layer untuk memproses fitur user
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(32, activation="relu")
        self.output_layer = Dense(embedding_dim, activation=None, name="user_vector")

    @tf.function  # Tambahkan untuk eksekusi dalam Graph Mode
    def call(self, inputs):
        # ✅ SOLUSI: Menggunakan slicing untuk menangani tensor input
        user_id = inputs[:, 0]  # Kolom pertama adalah user_id
        genre_pref = inputs[:, 1]  # Kolom kedua adalah genre (pastikan hanya 1 genre per user)

        # Embedding lookup
        user_emb = self.user_embedding(user_id)
        genre_emb = self.genre_embedding(genre_pref)

        # Flatten embeddings
        user_emb = Flatten()(user_emb)
        genre_emb = Flatten()(genre_emb)

        # Concatenate semua fitur
        x = Concatenate()([user_emb, genre_emb])
        x = self.dense1(x)
        x = self.dense2(x)

        return self.output_layer(x)
