import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten
from tensorflow.keras.models import Model

class TwoTowerModel(Model):
    def __init__(self, num_users, num_items, num_genres, num_authors, num_artists, embedding_dim, **kwargs):
        super(TwoTowerModel, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.num_genres = num_genres
        self.num_authors = num_authors
        self.num_artists = num_artists
        self.embedding_dim = embedding_dim

        # Embedding Layers
        self.user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)
        self.item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim)
        self.genre_embedding = Embedding(input_dim=num_genres, output_dim=embedding_dim)

        # Flatten layer
        self.flatten = Flatten()

        # Dense Layers
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_input, genre_input, item_input = inputs

        # Flattened Embeddings
        user_vec = self.flatten(self.user_embedding(user_input))
        item_vec = self.flatten(self.item_embedding(item_input))
        genre_vec = self.flatten(self.genre_embedding(genre_input))

        # Feature Interaction
        concat_vec = tf.concat([user_vec, genre_vec, item_vec], axis=1)
        x = self.dense1(concat_vec)
        x = self.dense2(x)
        output = self.output_layer(x)
    
        return output

    def get_config(self):
        config = super(TwoTowerModel, self).get_config()
        config.update({
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_genres": self.num_genres,
            "num_authors": self.num_authors,
            "num_artists": self.num_artists,
            "embedding_dim": self.embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
