import argparse
import pickle
from typing import Dict, List, Tuple
from functools import partial
import copy
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from tqdm import tqdm
import torch
from torch import optim
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, Subset
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss
import json
import os
from datetime import datetime
import argparse
import re

class DictDataSet(Dataset):
    def __init__(self, array_dict: Dict[str, np.ndarray]):
        self.keys_list = []
        for k, v in array_dict.items():
            self.keys_list.append(k)
            if np.issubdtype(v.dtype, np.dtype("bool")):
                setattr(self, k, torch.ByteTensor(v))
            elif np.issubdtype(v.dtype, np.int8):
                setattr(self, k, torch.CharTensor(v))
            elif np.issubdtype(v.dtype, np.int16):
                setattr(self, k, torch.ShortTensor(v))
            elif np.issubdtype(v.dtype, np.int32):
                setattr(self, k, torch.IntTensor(v))
            elif np.issubdtype(v.dtype, np.int64):
                setattr(self, k, torch.LongTensor(v))
            elif np.issubdtype(v.dtype, np.float32):
                setattr(self, k, torch.FloatTensor(v))
            elif np.issubdtype(v.dtype, np.float64):
                setattr(self, k, torch.DoubleTensor(v))
            else:
                setattr(self, k, torch.FloatTensor(v))

    def __getitem__(self, index):
        return {k: getattr(self, k)[index] for k in self.keys_list}

    def __len__(self):
        return getattr(self, self.keys_list[0]).shape[0]


def recycle(iterable):
    while True:
        for x in iterable:
            yield x


def get_set_and_loaders(
    data_dict: Dict[str, np.ndarray],
    shuffled_loader_config: Dict,
    serial_loader_config: Dict,
    ignore_keys: List[str] = None,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.DataLoader, torch.utils.data.DataLoader
]:
    dataset = DictDataSet(
        {k: v for k, v in data_dict.items() if (ignore_keys and k not in ignore_keys)}
    )
    loader = torch.utils.data.DataLoader(dataset, **shuffled_loader_config)
    serial_loader = torch.utils.data.DataLoader(dataset, **serial_loader_config)

    return dataset, iter(recycle(loader)), serial_loader


class QueueAggregator(object):
    def __init__(self, max_size):
        self._queued_list = []
        self.max_size = max_size

    def append(self, elem):
        self._queued_list.append(elem)
        if len(self._queued_list) > self.max_size:
            self._queued_list.pop(0)

    def get(self):
        return self._queued_list


def process_batch(
    batch: Dict[str, torch.tensor],
    model: nn.Module,
    quantiles_tensor: torch.tensor,
    device: torch.device,
):
    if is_cuda:
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch["target"]

    predicted_quantiles = batch_outputs["predicted_quantiles"]
    q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(
        outputs=predicted_quantiles, targets=labels, desired_quantiles=quantiles_tensor
    )
    return q_loss, q_risk

def aggregate_outputs(output_aggregator, validation_outputs = None):
    """
    Concatenate lists of arrays in output_aggregator into single arrays for each key.
    Returns a new dictionary in the same format as validation_outputs.
    """
    if validation_outputs is None:
        validation_outputs = dict()

    for k in list(output_aggregator.keys()):
        if len(output_aggregator[k]) == 0:
            continue
        new_output = np.concatenate(output_aggregator[k], axis=0)
        if k in validation_outputs:
            validation_outputs[k] = np.concatenate([validation_outputs[k], new_output], axis=0)
        else:
            validation_outputs[k] = new_output
    return validation_outputs

def predict(data_path, weights, file_path, cardinalities_map, start_from_checkpoint = False):
    print("predict open")

    with open(data_path, "rb") as fp:
        data = pickle.load(fp)
    print("data open")

    configuration = {
        "optimization": {
            "batch_size": {"training": 128, "inference": 128},  # both weere 64 before
            "learning_rate": 0.001,  # was 0.0001 april 1st 2025
            "max_grad_norm": 1,
        },
        "model": {
            "dropout": 0.1,  # was 0.05 before
            "state_size": 160,
            "output_quantiles": [0.1, 0.5, 0.9],
            "lstm_layers": 4,  # was 2
            "attention_heads": 4,  # was 4 #then 6
        },
        # these arguments are related to possible extensions of the model class
        "task_type": "regression",
        "target_window_start": None,
    }

    feature_map = data["feature_map"]
    # cardinalities_map = data['categorical_cardinalities']

    structure = {
        "num_historical_numeric": len(feature_map["historical_ts_numeric"]),
        "num_historical_categorical": len(feature_map["historical_ts_categorical"]),
        "num_static_numeric": len(feature_map["static_feats_numeric"]),
        "num_static_categorical": len(feature_map["static_feats_categorical"]),
        "num_future_numeric": len(feature_map["future_ts_numeric"]),
        "num_future_categorical": len(feature_map["future_ts_categorical"]),
        "historical_categorical_cardinalities": [
            cardinalities_map[feat] + 1
            for feat in feature_map["historical_ts_categorical"]
        ],
        "static_categorical_cardinalities": [
            cardinalities_map[feat] + 1
            for feat in feature_map["static_feats_categorical"]
        ],
        "future_categorical_cardinalities": [
            cardinalities_map[feat] + 1 for feat in feature_map["future_ts_categorical"]
        ],
    }

    configuration["data_props"] = structure
    model = TemporalFusionTransformer(config=OmegaConf.create(configuration))
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    model.to(device)
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        lr=configuration["optimization"]["learning_rate"],
    )

    shuffled_loader_config = {
        "batch_size": configuration["optimization"]["batch_size"]["training"],
        "drop_last": True,
        "shuffle": False,
    }

    serial_loader_config = {
        "batch_size": configuration["optimization"]["batch_size"]["inference"],
        "drop_last": False,
        "shuffle": False,
    }

    # the following fields do not contain actual data, but are only identifiers of each observation
    # meta_keys = ['time', 'location','soil', "soil_x", "soil_y", "id"]
    meta_keys = [
        "time",
        "location",
        "soil_x",
        "soil_y",
        "id",
    ]  #### CHANGED THIS since Boreal runs

    shuffled_loader_config = {"batch_size": 128, "drop_last": True, "shuffle": False}

    serial_loader_config = {"batch_size": 128, "drop_last": False, "shuffle": False}

    # the following fields do not contain actual data, but are only identifiers of each observation
    test_set, test_loader, test_serial_loader = get_set_and_loaders(
        data["data_sets"]["test"],
        serial_loader_config,
        serial_loader_config,
        ignore_keys=meta_keys,
    )

    model.load_state_dict(
        torch.load(weights, map_location=torch.device("cpu")), strict=False
    )
    model.eval()  # switch to evaluation mode

    output_aggregator = (
        dict()
    )  # will be used for aggregating the outputs across batches

    validation_outputs = dict()
    resume_batch_idx = 0  # zero-based index to start processing
    pred_folder = os.path.dirname(file_path)
    pred_filename = os.path.splitext(os.path.basename(file_path))[0]
        
    if start_from_checkpoint:
        # find the latest checkpoint file
        checkpoint_files = [
            f for f in os.listdir(pred_folder) if "checkpoint" in f and pred_filename in f and f.endswith('.pkl')
        ]
        batch_indices = []
        for fname in checkpoint_files:
            match = re.search(r"checkpoint_batch_(\d+)\.pkl", fname)
            if match:
                batch_indices.append(int(match.group(1)))

        if batch_indices:
            batch_indices = sorted(batch_indices)
            resume_batch_idx = batch_indices[-1]
            print("Resume at batch index:", resume_batch_idx)

            # aggregate outputs from all previous checkpoints
            loaded_per_key = dict()
            for b_idx in batch_indices:
                checkpoint_path = f"{pred_folder}/{pred_filename}_checkpoint_batch_{b_idx}.pkl"
                print(f"Loading checkpoint: {checkpoint_path}")
                with open(checkpoint_path, "rb") as cp_file:
                    checkpoint_data = pickle.load(cp_file)

                if not isinstance(checkpoint_data, dict):
                    raise TypeError(f"Checkpoint {checkpoint_path} did not contain a dict")

                for k, v in checkpoint_data.items():
                    loaded_per_key.setdefault(k, []).append(np.asarray(v))

            for k, list_of_arrays in loaded_per_key.items():
                try:
                    validation_outputs[k] = np.concatenate(list_of_arrays, axis=0)
                except Exception as e:
                    raise RuntimeError(f"Failed to concatenate loaded checkpoints for key '{k}': {e}")
        
    with torch.no_grad():
        total_batches = len(test_serial_loader)
        print("total batches", total_batches)
        checkpoint_interval = max(1, total_batches // 10) # Save every 10% of progress
        # go over the batches of the serial data loader
        
        checkpoint_output_aggregator = dict()

        for batch_idx, batch in enumerate(tqdm(
            test_serial_loader
        )):  # change this from validation serial loader
            # process each batch
            if batch_idx < resume_batch_idx and start_from_checkpoint:
                print(f"Skipping batch {batch_idx + 1} as it's already processed.")
                continue  # skip already processed batches  

            if is_cuda:
                for k in list(batch.keys()):
                    batch[k] = batch[k].to(device)
            batch_outputs = model(batch)

            # accumulate outputs, as well as labels
            for output_key, output_tensor in batch_outputs.items():
                checkpoint_output_aggregator.setdefault(output_key, []).append(
                    output_tensor.cpu().numpy()
                )

            # Checkpoint every 10% or at last batch
            if (batch_idx + 1) % checkpoint_interval == 0 or (batch_idx + 1) == total_batches:
                checkpoint_path = f"{pred_folder}/{pred_filename}_checkpoint_batch_{batch_idx+1}.pkl"
                print(f"Checkpoint at batch {batch_idx+1} to {checkpoint_path}")
                checkpoint_dict = aggregate_outputs(checkpoint_output_aggregator)
                with open(checkpoint_path, "wb") as cp_file:
                    pickle.dump(checkpoint_dict, cp_file)
                
                # After checkpointing, append to output_aggregator for final aggregation
                for k, v in checkpoint_output_aggregator.items():
                    output_aggregator.setdefault(k, []).extend(v)
                
                checkpoint_output_aggregator = dict()  # Reset for next checkpoint window
            
    validation_outputs = aggregate_outputs(output_aggregator, validation_outputs = validation_outputs)

    # Save the dictionary using Pickle
    with open(file_path, "wb") as pickle_file:
        print("saving in", file_path)
        pickle.dump(validation_outputs, pickle_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("--training_dir", type=str, help="data directory")
    parser.add_argument("--data_dir", type=str, help="data directory")
    parser.add_argument("--weights_dir", type=str, help="weight directory")
    parser.add_argument("--pred_dir", type=str, help="prediction directory")
    parser.add_argument(
        "--start_from_checkpoint", action="store_true", help="Start predicting from last checkpoint file"
    )
    args = parser.parse_args()

    training_directory = args.training_dir
    data_directory = args.data_dir
    weight_directory = args.weights_dir
    prediction_path = args.pred_dir
    start_from_checkpoint = args.start_from_checkpoint

    # only when predicting for 40 years
    # "/burg/glab/users/al4385/data/TFT_30_40years/"
    # data_l = data_directory.split("/")
    # join all the parts except the last two
    # data_d = "/".join(data_l[:-2])
    # add "TFT_30" to the path then add the last part of data_l
    # data_d =  "/burg/glab/users/al4385/data/TFT_30/merged_BDT_1982_2021.pkl"
    # print(data_d)
    with open(training_directory, "rb") as fp:
        data_o = pickle.load(fp)
    cardinalities_map = data_o["categorical_cardinalities"]
    # print("data done")
    # get today's date
    today = datetime.now()
    # format it like 2023-01-01
    today_str = today.strftime("%Y-%m-%d")

    weights = weight_directory
    file_path = prediction_path
    predict(data_directory, weights, file_path, cardinalities_map, start_from_checkpoint=start_from_checkpoint)
