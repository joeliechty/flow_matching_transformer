import os
import torch
import pickle


directory = "/home/joe/rss_latent_dynamics/metadata"

def inspect_pkl_file(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {filepath}")
    print("Type:", type(data))
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for k, v in data.items():
            print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
    elif isinstance(data, list):
        print(f"List of length {len(data)}")
        if len(data) > 0:
            print("First item type:", type(data[0]))
    else:
        print(data)


if __name__ == "__main__":
    filepath = os.path.join(directory, "push_cracker_nom_train_valid_data.pkl")
    inspect_pkl_file(filepath)
