import pandas as pd

class DataLoader:
    def __init__(self, user_interaction_path, item_metadata_path):
        self.user_interaction_path = user_interaction_path
        self.item_metadata_path = item_metadata_path

        # Load data
        self.user_data = pd.read_csv(self.user_interaction_path)
        self.item_data = pd.read_csv(self.item_metadata_path)

    def get_user_data(self):
        return self.user_data

    def get_item_data(self):
        return self.item_data

    def get_user_interactions(self, user_id):
        """ Mengembalikan semua interaksi user tertentu """
        return self.user_data[self.user_data['user_id'] == user_id]

    def get_item_metadata(self, book_id):
        """ Mengembalikan metadata dari item tertentu """
        return self.item_data[self.item_data['book_id'] == book_id]

# Contoh penggunaan
if __name__ == "__main__":
    data_loader = DataLoader("data/user_interactions.csv", "data/item_metadata.csv")

    # Cek data user
    print(data_loader.get_user_data().head())

    # Cek data item
    print(data_loader.get_item_data().head())
