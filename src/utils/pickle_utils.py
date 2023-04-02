
import pickle


def save(file_path, store_obj):
    with open(file_path, 'wb') as f:
        pickle.dump(store_obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_path):
    with open(file_path, 'rb') as f:
        stored_obj = pickle.load(f)
    return stored_obj

