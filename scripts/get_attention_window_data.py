#!/usr/bin/env python3
"""
Attention Window Data Extraction

For each sample at each phenological phase (SOS/EOS), finds the 30-day forecast
window with maximum total attention and saves its start index (position in the
historical time axis) to a CSV alongside geographic coordinates.

Output CSV schema:
    location | latitude | longitude | window_start

where window_start is the 0-based index into the attention time axis at which
the 30-day window of maximum summed attention begins.

Based on: Data/data_wrangling/get_driver_map_data_refactored.py
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AttentionWindowExtractor:
    """Extract the start index of the maximum-attention 30-day window from TFT predictions."""

    FORECAST_WINDOW = 30
    MAX_HISTORICAL_WINDOW = 365

    def __init__(
        self,
        data_path: str,
        pred_path: str,
        coords_path: str,
        sos_indices_path: Optional[str] = None,
        eos_indices_path: Optional[str] = None,
        output_dir: str = ".",
        pft: str = "UNKNOWN",
    ):
        self.data_path = Path(data_path)
        self.pred_path = Path(pred_path)
        self.coords_path = Path(coords_path)
        self.sos_indices_path = Path(sos_indices_path) if sos_indices_path else None
        self.eos_indices_path = Path(eos_indices_path) if eos_indices_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pft = pft

        self.data = None
        self.preds = None
        self.coords = None
        self.sos_indices = None
        self.eos_indices = None

        self.sos_window_df = None
        self.eos_window_df = None

    # ------------------------------------------------------------------
    # Validation & loading
    # ------------------------------------------------------------------

    def validate_paths(self) -> bool:
        missing = []
        for label, path in [
            ("Data file", self.data_path),
            ("Predictions file", self.pred_path),
            ("Coordinates file", self.coords_path),
        ]:
            if not path.exists():
                missing.append(f"{label}: {path}")

        if self.sos_indices_path and not self.sos_indices_path.exists():
            logger.warning(f"SOS indices not found: {self.sos_indices_path}")
        if self.eos_indices_path and not self.eos_indices_path.exists():
            logger.warning(f"EOS indices not found: {self.eos_indices_path}")

        if missing:
            for m in missing:
                logger.error(f"  Missing: {m}")
            return False

        logger.info("All required paths validated")
        return True

    def load_data(self) -> bool:
        try:
            logger.info(f"Loading data: {self.data_path}")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)

            logger.info(f"Loading predictions: {self.pred_path}")
            with open(self.pred_path, "rb") as f:
                self.preds = pickle.load(f)

            for key in ("attention_scores",):
                if key not in self.preds:
                    raise ValueError(f"Missing required prediction key: {key}")

            att_shape = self.preds["attention_scores"].shape
            logger.info(f"  attention_scores shape: {att_shape}")
            if len(att_shape) != 3:
                raise ValueError(f"Expected 3D attention array, got {att_shape}")

            logger.info(f"Loading coordinates: {self.coords_path}")
            if str(self.coords_path).endswith(".parquet"):
                self.coords = pd.read_parquet(self.coords_path)
            else:
                self.coords = pd.read_csv(self.coords_path)

            n_before = len(self.coords)
            self.coords = self.coords.drop_duplicates(subset=["location"], keep="first")
            logger.info(
                f"  Coordinates: {len(self.coords)} unique locations "
                f"({n_before - len(self.coords)} duplicates removed)"
            )

            if self.sos_indices_path and self.sos_indices_path.exists():
                self.sos_indices = np.load(self.sos_indices_path)
                logger.info(f"  Loaded {len(self.sos_indices)} SOS indices")

            if self.eos_indices_path and self.eos_indices_path.exists():
                self.eos_indices = np.load(self.eos_indices_path)
                logger.info(f"  Loaded {len(self.eos_indices)} EOS indices")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def find_max_attention_window(self, index: int) -> int:
        """
        Return the start index of the 30-day window with highest total attention.

        Averages attention across forecast horizons first, then slides a 30-day
        window over the historical time axis to find the position with maximum
        summed attention.

        Args:
            index: Sample index in the predictions array.

        Returns:
            0-based start index of the best window in the historical time axis.
        """
        att_array = np.mean(self.preds["attention_scores"][index], axis=0)
        max_start = min(336, len(att_array) - self.FORECAST_WINDOW)

        best_start = 0
        max_sum = -np.inf
        for i in range(max_start):
            s = np.sum(att_array[i : i + self.FORECAST_WINDOW])
            if s > max_sum:
                max_sum = s
                best_start = i

        return best_start

    def extract_window_starts(
        self, indices: np.ndarray, phase: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Find the max-attention window start for each sample in *indices*.

        Args:
            indices: Array of prediction indices (e.g. SOS or EOS test indices).
            phase:   "SOS" or "EOS" (used only for logging).

        Returns:
            DataFrame with columns [location, window_start] and a stats dict.
        """
        logger.info(f"Extracting {phase} attention window starts...")

        records = []
        stats = {
            "input_samples": len(indices),
            "skipped_bounds": 0,
            "skipped_location": 0,
            "errors": 0,
            "processed": 0,
        }

        max_window_start = self.MAX_HISTORICAL_WINDOW - self.FORECAST_WINDOW
        max_pred_index = len(self.preds["attention_scores"]) - 1

        for idx_num, index in enumerate(indices):
            try:
                if index > max_pred_index:
                    stats["skipped_bounds"] += 1
                    continue

                window_start = self.find_max_attention_window(index)

                if window_start > max_window_start:
                    stats["skipped_bounds"] += 1
                    continue

                try:
                    location_id = int(
                        self.data["data_sets"]["test"]["id"][index][0].split("_")[0]
                    )
                except (KeyError, IndexError, ValueError) as e:
                    stats["skipped_location"] += 1
                    logger.debug(f"Could not extract location for index {index}: {e}")
                    continue

                records.append({"location": location_id, "window_start": window_start})
                stats["processed"] += 1

                if (idx_num + 1) % 100 == 0:
                    logger.info(f"  {idx_num + 1}/{len(indices)} samples processed...")

            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"Error at index {index}: {e}")

        window_df = pd.DataFrame(records)

        logger.info(f"\n{phase} Window Extraction Results:")
        logger.info(f"  Input samples : {stats['input_samples']}")
        logger.info(f"  Processed     : {stats['processed']}")
        logger.info(f"  Skipped (bounds)  : {stats['skipped_bounds']}")
        logger.info(f"  Skipped (location): {stats['skipped_location']}")
        logger.info(f"  Errors        : {stats['errors']}")
        if len(window_df) > 0:
            logger.info(f"  Unique locations: {window_df['location'].nunique()}")

        return window_df, stats

    def merge_with_coordinates(
        self, window_df: pd.DataFrame, phase: str
    ) -> pd.DataFrame:
        """
        Aggregate duplicate locations (median of window_start) and join coordinates.

        Args:
            window_df: DataFrame with columns [location, window_start].
            phase:     "SOS" or "EOS" (used only for logging).

        Returns:
            DataFrame with columns [location, latitude, longitude, window_start].
        """
        logger.info(f"Merging {phase} windows with coordinates...")

        if len(window_df) > 0 and window_df["location"].duplicated().any():
            n_dup = window_df["location"].duplicated().sum()
            logger.info(
                f"  {n_dup} duplicate location entries — aggregating to unique locations with median"
            )
            window_df = window_df.groupby("location", as_index=False)[
                "window_start"
            ].median()
            # Round back to integer after median
            window_df["window_start"] = (
                window_df["window_start"].round().astype("Int64")
            )

        merged = pd.merge(self.coords, window_df, on="location", how="left")

        n_missing = merged["window_start"].isnull().sum()
        logger.info(f"  Total locations : {len(merged)}")
        logger.info(f"  With data       : {len(merged) - n_missing}")
        logger.info(
            f"  Missing         : {n_missing} ({100 * n_missing / len(merged):.1f}%)"
        )

        return merged

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def process_all(self) -> bool:
        logger.info("=" * 70)
        logger.info(f"ATTENTION WINDOW EXTRACTION - {self.pft}")
        logger.info("=" * 70)

        if not self.validate_paths():
            return False
        if not self.load_data():
            return False

        if self.sos_indices is None and self.eos_indices is None:
            logger.error(
                "No phenology indices provided (need --sos-indices and/or --eos-indices)."
            )
            return False

        if self.sos_indices is not None and len(self.sos_indices) > 0:
            df, _ = self.extract_window_starts(self.sos_indices, "SOS")
            self.sos_window_df = self.merge_with_coordinates(df, "SOS")

        if self.eos_indices is not None and len(self.eos_indices) > 0:
            df, _ = self.extract_window_starts(self.eos_indices, "EOS")
            self.eos_window_df = self.merge_with_coordinates(df, "EOS")

        logger.info("Processing complete")
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_diagnostics(self):
        """Histogram of window_start values and spatial scatter of coverage."""
        for df, phase, color in [
            (self.sos_window_df, "SOS", "green"),
            (self.eos_window_df, "EOS", "saddlebrown"),
        ]:
            if df is None or len(df) == 0:
                continue
            self._plot_window_distribution(df, phase, color)
            self._plot_spatial_coverage(df, phase, color)

    def _plot_window_distribution(self, df: pd.DataFrame, phase: str, color: str):
        data = df["window_start"].dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(
            data.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {data.mean():.1f}",
        )
        ax.axvline(
            data.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Median: {data.median():.0f}",
        )
        ax.set_title(
            f"{phase} Max-Attention Window Start Distribution — {self.pft}", fontsize=13
        )
        ax.set_xlabel("Window start index (days before forecast end)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.output_dir / f"{self.pft}_{phase}_window_distribution.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {out}")
        plt.close()

    def _plot_spatial_coverage(self, df: pd.DataFrame, phase: str, color: str):
        has_data = df["window_start"].notna()
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.scatter(
            df.loc[has_data, "longitude"],
            df.loc[has_data, "latitude"],
            c=df.loc[has_data, "window_start"],
            cmap="viridis",
            s=10,
            alpha=0.6,
            label=f"Has data ({has_data.sum()})",
        )
        ax.scatter(
            df.loc[~has_data, "longitude"],
            df.loc[~has_data, "latitude"],
            c="red",
            s=10,
            alpha=0.4,
            marker="x",
            label=f"Missing ({(~has_data).sum()})",
        )
        plt.colorbar(ax.collections[0], ax=ax, label="window_start index")
        ax.set_title(f"{phase} Max-Attention Window Start — {self.pft}", fontsize=13)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = self.output_dir / f"{self.pft}_{phase}_window_spatial.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {out}")
        plt.close()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_results(self, output_prefix: str):
        logger.info("Saving results...")
        for df, phase in [(self.sos_window_df, "SOS"), (self.eos_window_df, "EOS")]:
            if df is not None and len(df) > 0:
                path = self.output_dir / f"{output_prefix}_{phase}.csv"
                df.to_csv(path, index=False)
                logger.info(f"  Saved {phase}: {path}  shape={df.shape}")
        logger.info("=" * 70)
        logger.info("ATTENTION WINDOW EXTRACTION COMPLETE")
        logger.info("=" * 70)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract max-attention window start indices from TFT predictions"
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to processed data pickle"
    )
    parser.add_argument("--pred-path", required=True, help="Path to predictions pickle")
    parser.add_argument(
        "--coords-path", required=True, help="Path to coordinates CSV/parquet"
    )
    parser.add_argument("--sos-indices", help="Path to SOS indices numpy file")
    parser.add_argument("--eos-indices", help="Path to EOS indices numpy file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument(
        "--output-prefix", default="window_data", help="Prefix for output CSVs"
    )
    parser.add_argument(
        "--pft", default="UNKNOWN", help="Plant functional type identifier"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic plots")

    args = parser.parse_args()

    extractor = AttentionWindowExtractor(
        data_path=args.data_path,
        pred_path=args.pred_path,
        coords_path=args.coords_path,
        sos_indices_path=args.sos_indices,
        eos_indices_path=args.eos_indices,
        output_dir=args.output_dir,
        pft=args.pft,
    )

    if not extractor.process_all():
        logger.error("Processing failed.")
        return 1

    if not args.no_plots:
        extractor.plot_diagnostics()

    extractor.save_results(args.output_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
