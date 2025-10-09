import argparse
import pickle
import os
from typing import Dict, List
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import optim, nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss
import wandb


class DictDataSet(Dataset):
    """Dataset class for handling dictionary-based data."""

    def __init__(self, array_dict: Dict[str, np.ndarray]):
        self.keys_list = []
        for k, v in array_dict.items():
            self.keys_list.append(k)
            # Convert numpy arrays to appropriate torch tensors
            if np.issubdtype(v.dtype, np.dtype("bool")):
                setattr(self, k, torch.BoolTensor(v))
            elif np.issubdtype(v.dtype, np.int32):
                setattr(self, k, torch.IntTensor(v))
            elif np.issubdtype(v.dtype, np.int64):
                setattr(self, k, torch.LongTensor(v))
            elif np.issubdtype(v.dtype, np.float32):
                setattr(self, k, torch.FloatTensor(v))
            else:
                setattr(self, k, torch.FloatTensor(v))

    def __getitem__(self, index):
        return {k: getattr(self, k)[index] for k in self.keys_list}

    def __len__(self):
        return getattr(self, self.keys_list[0]).shape[0]


class QueueAggregator:
    """Running window aggregator for monitoring training performance."""

    def __init__(self, max_size):
        self._queued_list = []
        self.max_size = max_size
        self._cached_mean = None
        self._dirty = True

    def append(self, elem):
        self._queued_list.append(elem)
        if len(self._queued_list) > self.max_size:
            self._queued_list.pop(0)
        self._dirty = True

    def get(self):
        return self._queued_list

    def mean(self):
        if self._dirty or self._cached_mean is None:
            self._cached_mean = np.mean(self._queued_list) if self._queued_list else 0.0
            self._dirty = False
        return self._cached_mean


class EarlyStopping:
    """Early stopping mechanism."""

    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
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

        return self.num_bad_epochs >= self.patience

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")

        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            else:
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            else:
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def weight_init(m):
    """Initialize model weights."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def recycle(iterable):
    """Infinite iterator."""
    while True:
        for x in iterable:
            yield x


def get_data_loaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int,
    ignore_keys: List[str] = None,
    shuffle: bool = False,
):
    """Create dataset and data loaders."""
    filtered_dict = {
        k: v
        for k, v in data_dict.items()
        if ignore_keys is None or k not in ignore_keys
    }

    dataset = DictDataSet(filtered_dict)
    loader_config = {
        "batch_size": batch_size,
        "drop_last": True if shuffle else False,
        "shuffle": shuffle,
    }

    loader = DataLoader(dataset, **loader_config)
    return dataset, iter(recycle(loader)), loader


def process_batch(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    quantiles_tensor: torch.Tensor,
    device: torch.device,
):
    """Process a single batch."""
    # Move batch to device
    for k in batch.keys():
        batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch["target"]
    predicted_quantiles = batch_outputs["predicted_quantiles"]

    q_loss, q_risk, _ = tft_loss.get_quantiles_loss_and_q_risk(
        outputs=predicted_quantiles, targets=labels, desired_quantiles=quantiles_tensor
    )
    return q_loss, q_risk


def get_optimizer(model, lr_pretrained=1e-4, lr_new=1e-3, weight_decay=1e-4):
    """Create optimizer with different learning rates for different layer groups."""
    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(
                layer in name
                for layer in [
                    "categorical_embedding_layers",
                    "multihead_attn",
                    "output_layer",
                ]
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


def load_pretrained_model(model, checkpoint_path, device):
    """Load pretrained model weights."""
    print(f"Loading pre-trained checkpoint from {checkpoint_path}")
    pretrained_dict = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    # Filter out layers with size mismatches
    filtered_pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    # Log skipped layers
    for k in pretrained_dict.keys():
        if k not in filtered_pretrained_dict:
            print(f"Skipping layer due to size mismatch: {k}")

    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"Model loaded and moved to {device}.")


def setup_gradual_unfreezing(model, unfreeze_schedule, epoch_idx):
    """Handle gradual unfreezing of model layers."""
    if epoch_idx in unfreeze_schedule:
        print(f"Unfreezing layers at epoch {epoch_idx}")
        for name, param in model.named_parameters():
            if any(layer in name for layer in unfreeze_schedule[epoch_idx]):
                param.requires_grad = True
        return True
    return False


def evaluate_model(model, data_loaders, quantiles_tensor, device, eval_iters):
    """Evaluate model on all datasets."""
    model.eval()
    results = {}

    with torch.no_grad():
        for subset_name, subset_loader in data_loaders.items():
            q_loss_vals, q_risk_vals = [], []

            for _ in range(eval_iters):
                batch = next(subset_loader)
                batch_loss, batch_q_risk = process_batch(
                    batch, model, quantiles_tensor, device
                )
                q_loss_vals.append(batch_loss)
                q_risk_vals.append(batch_q_risk)

            eval_loss = torch.stack(q_loss_vals).mean(axis=0)
            eval_q_risk = torch.stack(q_risk_vals, axis=0).mean(axis=0)

            results[subset_name] = {"loss": eval_loss, "q_risk": eval_q_risk}

            print(
                f"Eval {subset_name} - q_loss = {eval_loss:.5f}, "
                + ", ".join(
                    [
                        f"q_risk_{q:.1f} = {risk:.5f}"
                        for q, risk in zip(quantiles_tensor, eval_q_risk)
                    ]
                )
            )

    return results


def main():
    # Configuration
    config = {
        "optimization": {
            "batch_size": {"training": 128, "inference": 128},
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
        },
        "model": {
            "dropout": 0.1,
            "state_size": 160,
            "output_quantiles": [0.1, 0.5, 0.9],
            "lstm_layers": 4,
            "attention_heads": 4,
        },
        "task_type": "regression",
        "target_window_start": None,
    }

    # Training hyperparameters
    training_config = {
        "max_epochs": 100,
        "log_interval": 50,
        "ma_queue_size": 50,
        "patience": 10,
    }

    # Unfreeze schedule
    unfreeze_schedule = {
        5: ["past_lstm", "future_lstm"],
        10: ["historical_ts_transform", "future_ts_transform"],
        15: ["static_transform", "static_selection"],
    }

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train TFT with transfer learning")
    parser.add_argument("--filename", type=str, help="data filename")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to pre-trained model"
    )
    parser.add_argument(
        "--data_directory",
        type=str,
        default="../../../glab/users/al4385/data/TFT_30/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--weights_directory",
        type=str,
        default="../../../glab/users/al4385/weights/pretrained_1219/",
        help="Path to weights directory",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Transfer_Learning_BET_freeze",
        help="wandb project name",
    )
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project=args.wandb_project, config=config)

    # Setup paths
    data_path = os.path.join(args.data_directory, args.filename)
    output_path = os.path.join(
        args.weights_directory, f"weights_{args.filename.split('.')[0]}_3.pth"
    )

    # Load data
    with open(data_path, "rb") as fp:
        data = pickle.load(fp)

    # Setup model structure
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
    config["data_props"] = structure

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TemporalFusionTransformer(config=OmegaConf.create(config))
    model.apply(weight_init)
    model.to(device)

    # Load pretrained weights if provided
    if args.checkpoint:
        load_pretrained_model(model, args.checkpoint, device)

    # Setup data loaders
    meta_keys = ["time", "location", "soil_x", "soil_y", "id"]
    batch_size_train = config["optimization"]["batch_size"]["training"]
    batch_size_inference = config["optimization"]["batch_size"]["inference"]

    train_set, train_loader, _ = get_data_loaders(
        data["data_sets"]["train"], batch_size_train, meta_keys, shuffle=False
    )
    val_set, val_loader, _ = get_data_loaders(
        data["data_sets"]["validation"], batch_size_train, meta_keys, shuffle=False
    )
    test_set, test_loader, _ = get_data_loaders(
        data["data_sets"]["test"], batch_size_inference, meta_keys, shuffle=False
    )

    data_loaders = {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }

    # Setup training parameters
    import math

    epoch_iters = len(data["data_sets"]["train"]["time_index"]) // batch_size_train
    eval_iters = math.ceil(
        len(data["data_sets"]["validation"]["time_index"]) / batch_size_train
    )

    # Initialize training components
    quantiles_tensor = torch.tensor(config["model"]["output_quantiles"]).to(device)
    es = EarlyStopping(patience=training_config["patience"])
    loss_aggregator = QueueAggregator(max_size=training_config["ma_queue_size"])

    # Freeze all layers initially except output layer
    for name, param in model.named_parameters():
        param.requires_grad = "output_layer" in name

    # Setup optimizer and scheduler
    opt = get_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )

    # Training loop
    batch_idx = 0
    for epoch_idx in range(training_config["max_epochs"]):
        print(f"Starting Epoch {epoch_idx}")

        # Handle gradual unfreezing
        if setup_gradual_unfreezing(model, unfreeze_schedule, epoch_idx):
            opt = get_optimizer(model)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=opt, mode="min", factor=0.5, patience=5, min_lr=1e-5
            )

        # Evaluation
        eval_results = evaluate_model(
            model, data_loaders, quantiles_tensor, device, eval_iters
        )
        validation_loss = eval_results["validation"]["loss"]

        wandb.log({"eval_q_loss": validation_loss})

        # Early stopping check
        if es.step(validation_loss):
            print("Performing early stopping!")
            break

        # Training
        model.train()
        for _ in range(epoch_iters):
            batch = next(train_loader)
            opt.zero_grad()

            loss, _ = process_batch(batch, model, quantiles_tensor, device)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            loss_aggregator.append(loss.item())

            if batch_idx % training_config["log_interval"] == 0:
                avg_loss = loss_aggregator.mean()
                print(
                    f"Epoch: {epoch_idx}, Batch: {batch_idx} - Train Loss = {avg_loss:.5f}"
                )

            batch_idx += 1

    print("Training completed. Saving model...")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    # Update scheduler and log final metrics
    scheduler.step(validation_loss)
    wandb.log(
        {
            "train_loss": loss_aggregator.mean(),
            "learning_rate": opt.param_groups[0]["lr"],
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
