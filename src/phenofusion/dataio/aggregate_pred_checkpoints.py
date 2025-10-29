#!/usr/bin/env python3
import argparse
import re
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

CHECKPOINT_RE = re.compile(r"(.+)_checkpoint_batch_(\d+)\.pkl$")


def find_checkpoint_files(folder: Path, base_filename: str):
    files = []
    for p in folder.iterdir():
        m = CHECKPOINT_RE.match(p.name)
        if m and m.group(1) == base_filename:
            files.append((int(m.group(2)), p))
    files.sort(key=lambda x: x[0])
    return [p for idx, p in files]


def load_and_concat(checkpoint_paths):
    per_key = {}
    for cp in tqdm(checkpoint_paths, desc="Loading checkpoints"):
        with open(cp, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Checkpoint {cp} did not contain a dict")
        for k, v in data.items():
            arr = np.asarray(v)
            per_key.setdefault(k, []).append(arr)
    # Concatenate per key
    out = {}
    for k, list_of_arr in per_key.items():
        try:
            out[k] = np.concatenate(list_of_arr, axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to concatenate key '{k}': {e}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_folder", required=True, help="Folder where checkpoints live"
    )
    parser.add_argument(
        "--pred_filename",
        required=True,
        help="base filename, e.g. BDT_-20_20_1982_2021.pkl",
    )
    parser.add_argument(
        "--out",
        required=False,
        help="final output path (defaults to pred_folder/pred_filename)",
        default=None,
    )
    args = parser.parse_args()

    folder = Path(args.pred_folder)
    base_name = Path(args.pred_filename).stem  # remove .pkl
    out_path = Path(args.out) if args.out else folder / args.pred_filename

    checkpoints = find_checkpoint_files(folder, base_name)
    if not checkpoints:
        print("No checkpoint files found.")
        return

    print(
        f"Found {len(checkpoints)} checkpoint files. First: {checkpoints[0].name}, Last: {checkpoints[-1].name}"
    )

    final_dict = load_and_concat(checkpoints)

    print(
        f"Saving final pickle to {out_path} ({sum(v.nbytes for v in final_dict.values()):,} bytes approx)"
    )
    with open(out_path, "wb") as f:
        pickle.dump(final_dict, f)


if __name__ == "__main__":
    main()
