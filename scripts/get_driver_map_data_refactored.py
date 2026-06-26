#!/usr/bin/env python3
"""
Driver Map Data Extraction - Refactored Version

This script extracts climate driver importance from TFT model predictions and creates
analysis-ready CSV files for driver map generation. It includes comprehensive diagnostics
and error handling.

Based on: src/phenofusion/dataio/driversdata.py
Author: Phenofusion Team
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DriverDataExtractor:
    """Extract driver importance data from TFT predictions."""

    DRIVER_NAMES = ["tmin", "tmax", "rad", "precip", "photo", "sm"]
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
        """
        Initialize the driver data extractor.

        Args:
            data_path: Path to processed data pickle file
            pred_path: Path to predictions pickle file
            coords_path: Path to coordinates CSV file
            sos_indices_path: Path to SOS indices numpy file (optional)
            eos_indices_path: Path to EOS indices numpy file (optional)
            output_dir: Directory for output files
            pft: Plant functional type identifier
        """
        self.data_path = Path(data_path)
        self.pred_path = Path(pred_path)
        self.coords_path = Path(coords_path)
        self.sos_indices_path = Path(sos_indices_path) if sos_indices_path else None
        self.eos_indices_path = Path(eos_indices_path) if eos_indices_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pft = pft

        # Data containers
        self.data = None
        self.preds = None
        self.coords = None
        self.sos_indices = None
        self.eos_indices = None

        # Results
        self.sos_driver_df = None
        self.eos_driver_df = None

    def validate_paths(self) -> bool:
        """Validate that all input files exist."""
        logger.info("Validating input paths...")

        missing = []
        if not self.data_path.exists():
            missing.append(f"Data file: {self.data_path}")
        if not self.pred_path.exists():
            missing.append(f"Predictions file: {self.pred_path}")
        if not self.coords_path.exists():
            missing.append(f"Coordinates file: {self.coords_path}")

        if self.sos_indices_path and not self.sos_indices_path.exists():
            logger.warning(f"SOS indices not found: {self.sos_indices_path}")
        if self.eos_indices_path and not self.eos_indices_path.exists():
            logger.warning(f"EOS indices not found: {self.eos_indices_path}")

        if missing:
            logger.error("Missing required files:")
            for path in missing:
                logger.error(f"  - {path}")
            return False

        logger.info("✓ All required paths validated")
        return True

    def load_data(self) -> bool:
        """Load all required data files."""
        logger.info("Loading data files...")

        try:
            # Load processed data
            logger.info(f"Loading data from: {self.data_path}")
            with open(self.data_path, "rb") as fp:
                self.data = pickle.load(fp)
            logger.info(f"  Loaded data with keys: {list(self.data.keys())}")

            # Load predictions
            logger.info(f"Loading predictions from: {self.pred_path}")
            with open(self.pred_path, "rb") as fp:
                self.preds = pickle.load(fp)
            logger.info(f"  Loaded predictions with keys: {list(self.preds.keys())}")

            # Validate prediction shapes
            self._validate_prediction_shapes()

            # Load coordinates
            logger.info(f"Loading coordinates from: {self.coords_path}")
            # Handle both CSV and Parquet formats
            if str(self.coords_path).endswith(".parquet"):
                self.coords = pd.read_parquet(self.coords_path)
            else:
                self.coords = pd.read_csv(self.coords_path)
            logger.info(f"  Loaded {len(self.coords)} coordinate records (raw)")

            # Deduplicate coordinates - keep only unique locations
            initial_size = len(self.coords)
            self.coords = self.coords.drop_duplicates(subset=["location"], keep="first")
            logger.info(
                f"  After deduplication: {len(self.coords)} unique locations ({initial_size - len(self.coords)} duplicates removed)"
            )
            logger.info(f"  Coordinate columns: {list(self.coords.columns)}")

            # Load phenology indices if available
            if self.sos_indices_path and self.sos_indices_path.exists():
                self.sos_indices = np.load(self.sos_indices_path)
                logger.info(f"  Loaded {len(self.sos_indices)} SOS indices")

            if self.eos_indices_path and self.eos_indices_path.exists():
                self.eos_indices = np.load(self.eos_indices_path)
                logger.info(f"  Loaded {len(self.eos_indices)} EOS indices")

            logger.info("✓ All data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return False

    def _validate_prediction_shapes(self):
        """Validate shapes of prediction arrays."""
        required_keys = ["historical_selection_weights", "attention_scores"]

        for key in required_keys:
            if key not in self.preds:
                raise ValueError(f"Missing required prediction key: {key}")

        hist_shape = self.preds["historical_selection_weights"].shape
        att_shape = self.preds["attention_scores"].shape

        logger.info(f"  Historical selection weights shape: {hist_shape}")
        logger.info(f"  Attention scores shape: {att_shape}")

        # Expected: (n_samples, time_steps, n_features)
        if len(hist_shape) != 3:
            raise ValueError(
                f"Expected 3D array for historical weights, got shape {hist_shape}"
            )

        if hist_shape[2] != 7:  # CSIF + 6 drivers
            logger.warning(
                f"Expected 7 features (CSIF + 6 drivers), got {hist_shape[2]}"
            )

    def find_max_attention_window(self, index: int) -> int:
        """
        Find the time window with maximum attention scores.

        Args:
            index: Sample index in prediction array

        Returns:
            Start index of the maximum attention window
        """
        # Get mean attention across all forecast horizons
        att_array = np.mean(self.preds["attention_scores"][index], axis=0)

        max_sum = -np.inf
        best_start_index = 0

        # Slide window to find maximum
        # Use 336 (365-30+1) as max_start to ensure we don't go beyond historical window
        max_start = min(336, len(att_array) - self.FORECAST_WINDOW)

        for i in range(max_start):
            current_sum = np.sum(att_array[i : i + self.FORECAST_WINDOW])
            if current_sum > max_sum:
                max_sum = current_sum
                best_start_index = i

        return best_start_index

    def extract_driver_weights(
        self, indices: np.ndarray, phase: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Extract driver importance weights for given phenology indices.

        Args:
            indices: Array of prediction indices
            phase: "SOS" or "EOS"

        Returns:
            DataFrame with driver weights and statistics dict
        """
        logger.info(f"Extracting {phase} driver weights...")

        driver_data = []
        stats = {
            "input_samples": len(indices),
            "skipped_window": 0,
            "skipped_location": 0,
            "errors": 0,
            "processed": 0,
        }

        max_window_start = self.MAX_HISTORICAL_WINDOW - self.FORECAST_WINDOW
        max_pred_index = len(self.preds["attention_scores"]) - 1

        for idx_num, index in enumerate(indices):
            try:
                # Validate index bounds
                if index > max_pred_index:
                    stats["skipped_window"] += 1
                    logger.debug(
                        f"Index {index} exceeds max prediction index {max_pred_index}"
                    )
                    continue

                # Find maximum attention window
                window_start = self.find_max_attention_window(index)

                if window_start > max_window_start:
                    stats["skipped_window"] += 1
                    logger.debug(
                        f"Window start {window_start} exceeds max {max_window_start}"
                    )
                    continue

                # Extract weights for each driver
                weights = {}
                hist_weights = self.preds["historical_selection_weights"][index]

                for i, var in enumerate(
                    self.DRIVER_NAMES, 1
                ):  # Start at 1 (skip CSIF at index 0)
                    weight_values = hist_weights[
                        window_start : window_start + self.FORECAST_WINDOW, i
                    ]
                    weights[f"hist_{var}"] = np.median(weight_values)

                # Extract location ID
                try:
                    location_id = int(
                        self.data["data_sets"]["test"]["id"][index][0].split("_")[0]
                    )
                    weights["location"] = location_id
                except (KeyError, IndexError, ValueError) as e:
                    stats["skipped_location"] += 1
                    logger.debug(f"Could not extract location for index {index}: {e}")
                    continue

                driver_data.append(weights)
                stats["processed"] += 1

                # Progress logging
                if (idx_num + 1) % 100 == 0:
                    logger.info(f"  Processed {idx_num + 1}/{len(indices)} samples...")

            except Exception as e:
                stats["errors"] += 1
                logger.debug(f"Error processing index {index}: {e}")
                continue

        # Create DataFrame
        driver_df = pd.DataFrame(driver_data)

        # Log results
        logger.info(f"\n{phase} Driver Extraction Results:")
        logger.info(f"  Input samples: {stats['input_samples']}")
        logger.info(f"  Successfully processed: {stats['processed']}")
        logger.info(f"  Skipped (window): {stats['skipped_window']}")
        logger.info(f"  Skipped (location): {stats['skipped_location']}")
        logger.info(f"  Errors: {stats['errors']}")

        if len(driver_df) > 0:
            logger.info(f"  DataFrame shape: {driver_df.shape}")
            logger.info(f"  Unique locations: {driver_df['location'].nunique()}")
        else:
            logger.warning(f"  No data extracted for {phase}!")

        return driver_df, stats

    def merge_with_coordinates(
        self, driver_df: pd.DataFrame, phase: str
    ) -> pd.DataFrame:
        """
        Merge driver weights with coordinate information.

        Args:
            driver_df: DataFrame with driver weights
            phase: "SOS" or "EOS"

        Returns:
            Merged DataFrame with coordinates
        """
        logger.info(f"Merging {phase} drivers with coordinates...")

        merged_df = pd.merge(self.coords, driver_df, on="location", how="left")

        driver_cols = [f"hist_{var}" for var in self.DRIVER_NAMES]
        missing_mask = merged_df[driver_cols].isnull().all(axis=1)
        n_missing = missing_mask.sum()

        logger.info(f"  Total locations: {len(merged_df)}")
        logger.info(f"  Locations with data: {len(merged_df) - n_missing}")
        logger.info(
            f"  Locations missing all drivers: {n_missing} ({100*n_missing/len(merged_df):.1f}%)"
        )

        return merged_df

    def process_all(self) -> bool:
        """Execute full processing pipeline."""
        logger.info("=" * 70)
        logger.info(f"DRIVER DATA EXTRACTION - {self.pft}")
        logger.info("=" * 70)

        # Validate and load
        if not self.validate_paths():
            return False

        if not self.load_data():
            return False

        # Check if we have phenology indices
        if self.sos_indices is None and self.eos_indices is None:
            logger.error("No phenology indices provided! Need SOS and/or EOS indices.")
            return False

        # Process SOS
        if self.sos_indices is not None and len(self.sos_indices) > 0:
            self.sos_driver_df, sos_stats = self.extract_driver_weights(
                self.sos_indices, "SOS"
            )
            self.sos_driver_df = self.merge_with_coordinates(self.sos_driver_df, "SOS")

        # Process EOS
        if self.eos_indices is not None and len(self.eos_indices) > 0:
            self.eos_driver_df, eos_stats = self.extract_driver_weights(
                self.eos_indices, "EOS"
            )
            self.eos_driver_df = self.merge_with_coordinates(self.eos_driver_df, "EOS")

        logger.info("✓ Processing complete")
        return True

    def plot_diagnostics(self):
        """Generate diagnostic plots."""
        logger.info("Generating diagnostic plots...")

        driver_cols = [f"hist_{var}" for var in self.DRIVER_NAMES]

        # Plot driver weight distributions
        for df, phase, color in [
            (self.sos_driver_df, "SOS", "green"),
            (self.eos_driver_df, "EOS", "brown"),
        ]:
            if df is None or len(df) == 0:
                continue

            self._plot_driver_distributions(df, phase, color, driver_cols)
            self._plot_spatial_coverage(df, phase, color, driver_cols)
            self._plot_driver_correlations(df, phase, driver_cols)

    def _plot_driver_distributions(
        self, df: pd.DataFrame, phase: str, color: str, driver_cols: List[str]
    ):
        """Plot distribution of driver weights."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, col in enumerate(driver_cols):
            data = df[col].dropna()
            if len(data) == 0:
                continue

            axes[idx].hist(data, bins=50, alpha=0.7, color=color, edgecolor="black")
            axes[idx].axvline(
                data.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {data.mean():.4f}",
            )
            axes[idx].axvline(
                data.median(),
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Median: {data.median():.4f}",
            )
            axes[idx].set_title(f'{phase} - {col.replace("hist_", "").upper()}')
            axes[idx].set_xlabel("Weight")
            axes[idx].set_ylabel("Frequency")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(
            f"{phase} Driver Weight Distributions - {self.pft}", fontsize=16, y=1.00
        )
        plt.tight_layout()

        output_path = self.output_dir / f"{self.pft}_{phase}_driver_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def _plot_spatial_coverage(
        self, df: pd.DataFrame, phase: str, color: str, driver_cols: List[str]
    ):
        """Plot spatial coverage map."""
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

        has_data = ~df[driver_cols].isnull().all(axis=1)

        ax.scatter(
            df.loc[has_data, "longitude"],
            df.loc[has_data, "latitude"],
            c=color,
            s=20,
            alpha=0.6,
            label=f"Has data ({has_data.sum()} locations)",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.scatter(
            df.loc[~has_data, "longitude"],
            df.loc[~has_data, "latitude"],
            c="red",
            s=20,
            alpha=0.6,
            label=f"Missing ({(~has_data).sum()} locations)",
            marker="x",
        )

        ax.set_title(
            f"{phase} Spatial Coverage - {self.pft}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / f"{self.pft}_{phase}_spatial_coverage.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def _plot_driver_correlations(
        self, df: pd.DataFrame, phase: str, driver_cols: List[str]
    ):
        """Plot correlation matrix of drivers."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Compute correlation matrix
        corr_matrix = df[driver_cols].corr()

        # Plot heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation"},
            ax=ax,
        )

        # Clean up labels
        labels = [col.replace("hist_", "").upper() for col in driver_cols]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)
        ax.set_title(
            f"{phase} Driver Correlations - {self.pft}", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        output_path = self.output_dir / f"{self.pft}_{phase}_driver_correlations.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def save_results(self, output_prefix: str):
        """Save extracted driver data to CSV files."""
        logger.info("Saving results...")

        # Save SOS
        if self.sos_driver_df is not None and len(self.sos_driver_df) > 0:
            sos_path = self.output_dir / f"{output_prefix}_SOS.csv"
            self.sos_driver_df.to_csv(sos_path, index=False)
            logger.info(f"  ✓ SOS data saved: {sos_path}")
            logger.info(f"    Shape: {self.sos_driver_df.shape}")

        # Save EOS
        if self.eos_driver_df is not None and len(self.eos_driver_df) > 0:
            eos_path = self.output_dir / f"{output_prefix}_EOS.csv"
            self.eos_driver_df.to_csv(eos_path, index=False)
            logger.info(f"  ✓ EOS data saved: {eos_path}")
            logger.info(f"    Shape: {self.eos_driver_df.shape}")

        logger.info("=" * 70)
        logger.info("DRIVER DATA EXTRACTION COMPLETE!")
        logger.info("=" * 70)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract climate driver importance from TFT predictions"
    )

    parser.add_argument(
        "--data-path", required=True, help="Path to processed data pickle file"
    )
    parser.add_argument(
        "--pred-path", required=True, help="Path to predictions pickle file"
    )
    parser.add_argument(
        "--coords-path", required=True, help="Path to coordinates CSV file"
    )
    parser.add_argument("--sos-indices", help="Path to SOS indices numpy file")
    parser.add_argument("--eos-indices", help="Path to EOS indices numpy file")
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for results and plots"
    )
    parser.add_argument(
        "--output-prefix", default="driver_data", help="Prefix for output CSV files"
    )
    parser.add_argument(
        "--pft", default="UNKNOWN", help="Plant functional type identifier"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating diagnostic plots"
    )

    args = parser.parse_args()

    # Create extractor
    extractor = DriverDataExtractor(
        data_path=args.data_path,
        pred_path=args.pred_path,
        coords_path=args.coords_path,
        sos_indices_path=args.sos_indices,
        eos_indices_path=args.eos_indices,
        output_dir=args.output_dir,
        pft=args.pft,
    )

    # Process
    success = extractor.process_all()

    if not success:
        logger.error("Processing failed!")
        return 1

    # Generate diagnostics
    if not args.no_plots:
        extractor.plot_diagnostics()

    # Save results
    extractor.save_results(args.output_prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())
