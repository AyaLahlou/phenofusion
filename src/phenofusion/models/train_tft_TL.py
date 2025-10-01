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
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[: n // 3].fill_(-1.0)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


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


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


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


if __name__ == "__main__":
    wandb.init(
        project="Transfer_Learning_BET_freeze",
        config={
            "batch_size": {"training": 128, "inference": 128},  # both weere 64 before
            "learning_rate": 0.001,  # was 0.001
            "max_grad_norm": 1.0,
            "dropout": 0.1,  # was 0.05 before
            "state_size": 160,
            "output_quantiles": [0.1, 0.5, 0.9],
            "lstm_layers": 4,  # was 2ß
            "attention_heads": 4,  # was 4 #then 6
        },
    )

    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("filename", type=str, help="data filename")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to pre-trained model"
    )

    args = parser.parse_args()
    filename = args.filename

    configuration = {
        "optimization": {
            "batch_size": {"training": 128, "inference": 128},  # both weere 64 before
            "learning_rate": 0.001,  # was 0.001
            "max_grad_norm": 1.0,
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

    data_directory = "../../../glab/users/al4385/data/TFT_30/"
    weights_directory = "../../../glab/users/al4385/weights/pretrained_1219/"
    # for filename in os.listdir(data_directory):
    # if os.path.isfile(os.path.join(data_directory, filename)):
    data_path = os.path.join(data_directory, filename)
    print(data_path)
    output_path = os.path.join(
        weights_directory, "weights_" + filename.split(".")[0] + "_3.pth"
    )
    print(output_path)
    with open(data_path, "rb") as fp:
        data = pickle.load(fp)

    feature_map = data["feature_map"]
    cardinalities_map = data["categorical_cardinalities"]

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
    model.apply(weight_init)
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    # Check CUDA availability
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))

    print(device)
    model.to(device)
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, list(model.parameters())),
        lr=configuration["optimization"]["learning_rate"],
        weight_decay=1e-4,
    )

    # Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", factor=0.5, patience=5, min_lr=1e-6
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
    meta_keys = ["time", "location", "soil_x", "soil_y", "id"]
    train_set, train_loader, train_serial_loader = get_set_and_loaders(
        data["data_sets"]["train"],
        shuffled_loader_config,
        serial_loader_config,
        ignore_keys=meta_keys,
    )
    validation_set, validation_loader, validation_serial_loader = get_set_and_loaders(
        data["data_sets"]["validation"],
        shuffled_loader_config,
        serial_loader_config,
        ignore_keys=meta_keys,
    )
    test_set, test_loader, test_serial_loader = get_set_and_loaders(
        data["data_sets"]["test"],
        serial_loader_config,
        serial_loader_config,
        ignore_keys=meta_keys,
    )

    # If early stopping is not triggered, after how many epochs should we quit training
    max_epochs = 100
    # how many training batches will compose a single training epoch
    epoch_iters = (
        len(data["data_sets"]["train"]["time_index"]) // 128
    )  # was 200 #then 400
    # upon completing a training epoch, we perform an evaluation of all the subsets
    # eval_iters will define how many batches of each set will compose a single evaluation round
    eval_iters = (
        len(data["data_sets"]["validation"]["time_index"]) // 128
    )  # 500 #then 100
    # during training, on what frequency should we display the monitored performance
    log_interval = 50
    # what is the running-window used by our QueueAggregator object for monitoring the training performance
    ma_queue_size = 50
    # how many evaluation rounds should we allow,
    # without any improvement in the performance observed on the validation set
    patience = 10

    # initialize early stopping mechanism
    es = EarlyStopping(patience=patience)
    # initialize the loss aggregator for running window performance estimation
    loss_aggregator = QueueAggregator(max_size=ma_queue_size)

    # initialize counters
    batch_idx = 0
    epoch_idx = 0

    quantiles_tensor = torch.tensor(configuration["model"]["output_quantiles"]).to(
        device
    )

    if args.checkpoint:
        print(f"Loading pre-trained checkpoint from {args.checkpoint}")
        pretrained_dict = torch.load(args.checkpoint, map_location=device)
        print("Model layers:")
        for name, param in model.named_parameters():
            print(name)

        print("pretrained layers:")
        for name, param in pretrained_dict.items():
            print(name)

        model_dict = model.state_dict()

        print("Model layers:")
        for name, param in model_dict.items():
            print(name)

        # Filter out unnecessary keys
        filtered_pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }

        for k in pretrained_dict.keys():
            if k not in filtered_pretrained_dict:
                print(f"Skipping layer due to size mismatch: {k}")

        # Overwrite entries in the existing state dict
        model_dict.update(filtered_pretrained_dict)

        # Load the new state dict
        model.load_state_dict(model_dict, strict=False)
        # Move model to device
        model.to(device)
        print(f"Model loaded and moved to {device}.")

    # Define unfreeze schedule
    unfreeze_schedule = {
        5: ["past_lstm", "future_lstm"],  # Unfreeze LSTMs at epoch 5
        10: [
            "historical_ts_transform",
            "future_ts_transform",
        ],  # Unfreeze transforms at epoch 10
        15: [
            "static_transform",
            "static_selection",
        ],  # Unfreeze static-related layers at epoch 15
    }

    # Freeze all layers initially
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "output_layer" in name:  # Replace with your output layer name
            param.requires_grad = True

    # Print which layers are frozen
    print("Initial layer freeze status:")
    for name, param in model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    # Initialize optimizer and scheduler
    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Group layers based on whether they're newly initialized or pretrained
            if (
                "categorical_embedding_layers" in name
                or "multihead_attn" in name
                or "output_layer" in name
            ):
                new_params.append(param)  # Newly initialized layers
            else:
                pretrained_params.append(param)  # Pretrained layers

    # Initialize optimizer and scheduler
    def get_optimizer(model, lr_pretrained=1e-4, lr_new=1e-3, weight_decay=1e-4):
        pretrained_params = []
        new_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if (
                    "categorical_embedding_layers" in name
                    or "multihead_attn" in name
                    or "output_layer" in name
                ):
                    new_params.append(param)
                else:
                    pretrained_params.append(param)

        optimizer = optim.Adam(
            [
                {"params": pretrained_params, "lr": lr_pretrained},
                {"params": new_params, "lr": lr_new},
            ],
            weight_decay=weight_decay,
        )
        return optimizer

    # Optimizer with separate learning rates
    opt = get_optimizer(model)  # Optimizer initialization function

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    epoch_idx = 0

    while epoch_idx < max_epochs:
        print(f"Starting Epoch Index {epoch_idx}")

        ## Gradual unfreezing
        if epoch_idx in unfreeze_schedule:
            print(f"Unfreezing layers at epoch {epoch_idx}")
            for name, param in model.named_parameters():
                if any(layer in name for layer in unfreeze_schedule[epoch_idx]):
                    param.requires_grad = True

            # Update optimizer with new trainable parameters
            opt = get_optimizer(model)  # Recreate optimizer if trainable params change
            # IMPORTANT: Do NOT recreate the scheduler unless optimizer changes
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt, mode="min", factor=0.5, patience=5, min_lr=1e-5
            )

        # Print updated layer status
        print("Updated layer freeze status:")
        for name, param in model.named_parameters():
            print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

        # evaluation round
        model.eval()
        with torch.no_grad():
            # for each subset
            for subset_name, subset_loader in zip(
                ["train", "validation", "test"],
                [train_loader, validation_loader, test_loader],
            ):
                print(f"Evaluating {subset_name} set")

                q_loss_vals, q_risk_vals = (
                    [],
                    [],
                )  # used for aggregating performance along the evaluation round
                for v in range(eval_iters):
                    # print(v)
                    # get batch
                    batch = next(subset_loader)
                    # batch = [item.to(device) for item in batch]
                    # process batch
                    batch_loss, batch_q_risk = process_batch(
                        batch=batch,
                        model=model,
                        quantiles_tensor=quantiles_tensor,
                        device=device,
                    )
                    # accumulate performance
                    q_loss_vals.append(batch_loss)
                    q_risk_vals.append(batch_q_risk)
                # print('done')
                # aggregate and average
                eval_loss = torch.stack(q_loss_vals).mean(axis=0)
                eval_q_risk = torch.stack(q_risk_vals, axis=0).mean(axis=0)

                # keep for feeding the early stopping mechanism
                if subset_name == "validation":
                    validation_loss = eval_loss

                # log performance
                print(
                    f"Epoch: {epoch_idx}, Batch Index: {batch_idx}"
                    + f"- Eval {subset_name} - "
                    + f"q_loss = {eval_loss:.5f} , "
                    + " , ".join(
                        [
                            f"q_risk_{q:.1} = {risk:.5f}"
                            for q, risk in zip(quantiles_tensor, eval_q_risk)
                        ]
                    )
                )

                # log metrics to wandb
                wandb.log({"eval_q_loss": eval_loss})
        # switch to training mode
        model.train()

        # update early stopping mechanism and stop if triggered
        if es.step(validation_loss):
            print("Performing early stopping...!")
            break

        # initiating a training round
        for _ in range(epoch_iters):
            # get training batch
            batch = next(train_loader)

            opt.zero_grad()
            # process batch
            loss, _ = process_batch(
                batch=batch,
                model=model,
                quantiles_tensor=quantiles_tensor,
                device=device,
            )

            # Debug gradient norms
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            # compute gradients
            loss.backward()
            # update weights
            opt.step()
            # gradient clipping

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # accumulate performance
            loss_aggregator.append(loss.item())

            # log performance
            if batch_idx % log_interval == 0:
                print(
                    f"Epoch: {epoch_idx}, Batch Index: {batch_idx} - Train Loss = {np.mean(loss_aggregator.get())}"
                )

            # completed batch
            batch_idx += 1

        scheduler.step(validation_loss)

        # Log the learning rate
        current_lr = opt.param_groups[0]["lr"]

        wandb.log({"learning_rate": current_lr})

        # log metrics to wandb
        wandb.log({"train_loss": np.mean(loss_aggregator.get())})

        # completed epoch
        epoch_idx += 1

    # Save model weights to a file
    print("Training over.")
    torch.save(model.state_dict(), output_path)
    wandb.finish()
