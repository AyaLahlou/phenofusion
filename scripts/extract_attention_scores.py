#!/usr/bin/env python3
"""
Extract attention_scores from a pickle file and save to a new file.
Uses memory-efficient approach for large files.
"""

import pickle
import gc

# Input and output paths
input_path = "/burg/glab/users/al4385/predictions/pred_40year_moresamples/BDT_50_20_1982_2021.pkl"
output_path = "/burg/glab/users/al4385/predictions/pred_40year_moresamples/BDT_50_20_1982_2021_attention_only.pkl"


# Custom unpickler to selectively load data
class SelectiveUnpickler(pickle.Unpickler):
    def __init__(self, file, key_to_keep):
        super().__init__(file)
        self.key_to_keep = key_to_keep
        self.result = {}

    def persistent_load(self, pid):
        # Handle persistent IDs if needed
        return None


print(f"Loading pickle from: {input_path}")
print("This may take a while for large files...")

# Try to load with pickle protocol
with open(input_path, "rb") as f:
    try:
        # First, peek at the structure
        data = pickle.load(f)
        print(f"\nOriginal file has {len(data)} keys: {list(data.keys())}")

        # Extract only attention_scores
        attention_scores = data["attention_scores"]
        print(f"Extracted attention_scores with shape: {attention_scores.shape}")

        # Clear the full data from memory
        keys_to_delete = [k for k in data.keys() if k != "attention_scores"]
        for key in keys_to_delete:
            del data[key]
        del data
        gc.collect()

        # Save to new file
        attention_data = {"attention_scores": attention_scores}
        print(f"\nSaving to: {output_path}")
        with open(output_path, "wb") as out_f:
            pickle.dump(attention_data, out_f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Done!")

    except MemoryError:
        print("\nMemoryError: File is too large to load completely.")
        print("You may need to run this on a high-memory node with more RAM.")
