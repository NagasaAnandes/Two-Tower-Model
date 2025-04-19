import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Concatenate
from tensorflow.keras.models import Model

class ItemTower(Model):
    def __init__(self, num_genres, num_authors, num_artists, embedding_dim=32):
        super(ItemTower, self).__init__()

        # Embedding untuk genre, penulis, artist
        self.genre_embedding = Embedding(input_dim=num_genres, output_dim=embedding_dim, name="genre_embedding")
        self.author_embedding = Embedding(input_dim=num_authors, output_dim=embedding_dim, name="author_embedding")
        self.artist_embedding = Embedding(input_dim=num_artists, output_dim=embedding_dim, name="artist_embedding")

        # Dense layer untuk memproses semua fitur
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(32, activation="relu")
        self.output_layer = Dense(embedding_dim, activation=None, name="item_vector")

    def call(self, inputs):
        genre_id, author_id, artist_id = inputs

        # Embedding lookup
        genre_emb = self.genre_embedding(genre_id)
        author_emb = self.author_embedding(author_id)
        artist_emb = self.artist_embedding(artist_id)

        # Flatten embeddings
        genre_emb = Flatten()(genre_emb)
        author_emb = Flatten()(author_emb)
        artist_emb = Flatten()(artist_emb)

        # Concatenate semua fitur
        x = Concatenate()([genre_emb, author_emb, artist_emb])
        x = self.dense1(x)
        x = self.dense2(x)

        return self.output_layer(x)
