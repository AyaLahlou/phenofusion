"""
REFACTORED SOS/EOS DETECTION WITH PHENOLOGICAL CORRECTION
==========================================================

This script detects Start of Season (SOS) and End of Season (EOS) samples
from forecast batch data, with proper phenological stage validation.

Key improvements:
- Works for both BDT_50_20 and BDT_50_90 datasets
- Adds CSIF threshold checks to ensure same phenological stage
- Clean, modular code structure
- Comprehensive validation and visualization
- Production-ready with error handling

Author: Refactored from original notebook
Date: 2026-01-08
"""

import json
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """Configuration parameters for SOS/EOS detection"""

    # Dataset selection
    PFT: str = "BDT_50_90"  # Options: "BDT_50_20", "BDT_50_90"

    # Paths
    data_dir: str = "/burg/glab/users/al4385/ms7073/analysis/driversdata/oversampling/"
    config_dir: str = (
        "/burg-archive/home/al4385/phenofusion/src/phenofusion/debugging/sos_eos_json/"
    )
    output_dir: str = "/burg-archive/home/al4385/phenofusion/src/phenofusion/debugging/sos_eos_indices_pickle/"

    # Geographic bounds
    lat_min: float = 45.0
    lat_max: float = 60.0
    lon_min: float = -10.0
    lon_max: float = 40.0

    # Batch parameters
    forecast_window: int = 30

    # Phenological thresholds (KEY PARAMETERS FOR FIX)
    csif_threshold_sos: float = 0.0  # Minimum mean CSIF for SOS
    csif_threshold_eos: float = 0.0  # Minimum mean CSIF for EOS
    min_csif_increase_sos: float = 0.05  # Minimum CSIF increase across batch
    max_csif_decrease_eos: float = -0.05  # Maximum CSIF decrease across batch

    # Validation
    create_plots: bool = True
    plot_examples: int = 30  # Number of example batches to plot

    def get_data_file(self) -> str:
        """Get the data file path for current PFT"""
        return f"{self.data_dir}{self.PFT}_analysisdf.csv"

    def get_config_file(self) -> str:
        """Get the config file path for current PFT"""
        return f"{self.config_dir}{self.PFT}_phenology_config.json"

    def get_output_file(self, corrected: bool = True) -> str:
        """Get the output pickle file path"""
        suffix = "_CORRECTED" if corrected else ""
        return f"{self.output_dir}subset_{self.PFT}_sos_eos_dict{suffix}.pkl"


# ============================================================================
# DATA LOADING
# ============================================================================


class DataLoader:
    """Handle data loading and preprocessing"""

    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the analysis dataframe"""
        logger.info(f"Loading data from {self.config.get_data_file()}")

        df = pd.read_csv(self.config.get_data_file())

        # Apply geographic bounds
        df = df[
            (df["latitude"].between(self.config.lat_min, self.config.lat_max))
            & (df["longitude"].between(self.config.lon_min, self.config.lon_max))
        ]

        # Convert time and sort
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(by=["location", "time"], ascending=[True, True])

        # Remove NaNs
        df = df.dropna()

        logger.info(f"✓ Loaded {len(df):,} records")
        logger.info(f"  Unique locations: {df['location'].nunique():,}")
        logger.info(
            f"  Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}"
        )
        logger.info(f"  DOY range: {df['doy'].min()} to {df['doy'].max()}")
        logger.info(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def load_config(self) -> Dict:
        """Load the phenology configuration"""
        config_file = self.config.get_config_file()
        logger.info(f"Loading config from {config_file}")

        with open(config_file, "r") as f:
            cfg = json.load(f)

        logger.info(f"✓ Config loaded for {self.config.PFT}")

        return cfg


# ============================================================================
# PHENOLOGY DETECTION
# ============================================================================


@dataclass
class DetectionResults:
    """Container for detection results"""

    SOS_indices: List[int]
    EOS_indices: List[int]
    sos_stats: Dict
    eos_stats: Dict
    skipped_stats: Dict


class PhenologyDetector:
    """Detect SOS and EOS samples with phenological validation"""

    def __init__(self, config: Config, df: pd.DataFrame, phenology_config: Dict):
        self.config = config
        self.df = df
        self.phenology_config = phenology_config

    def detect(self) -> DetectionResults:
        """Main detection loop"""
        logger.info("=" * 80)
        logger.info("PHENOLOGY DETECTION WITH CORRECTION")
        logger.info("=" * 80)
        logger.info("Parameters:")
        logger.info(f"  CSIF threshold (SOS): {self.config.csif_threshold_sos}")
        logger.info(f"  Min CSIF increase (SOS): {self.config.min_csif_increase_sos}")
        logger.info(f"  Batch size: {self.config.forecast_window}")

        # Initialize
        SOS_indices = []
        EOS_indices = []

        sos_starts = []
        eos_starts = []
        sos_csif_means = []
        eos_csif_means = []
        sos_csif_increases = []
        eos_csif_decreases = []

        # Counters
        counters = {
            "sos_count": 0,
            "eos_count": 0,
            "skipped_incomplete": 0,
            "skipped_multi_location": 0,
            "skipped_sos_monotonic": 0,
            "skipped_eos_monotonic": 0,
            "skipped_sos_csif_threshold": 0,
            "skipped_eos_csif_threshold": 0,
            "skipped_sos_weak_signal": 0,
            "skipped_eos_weak_signal": 0,
            "skipped_nan": 0,
            "skipped_no_config": 0,
        }

        batch_size = self.config.forecast_window
        total_batches = len(self.df) // batch_size

        logger.info(
            f"\nScanning {len(self.df):,} records in batches of {batch_size}..."
        )
        logger.info(f"Total batches: {total_batches:,}\n")

        # Progress reporting
        report_interval = max(1000, total_batches // 20)

        # Process batches
        for batch_num, start in enumerate(range(0, len(self.df), batch_size)):
            if batch_num % report_interval == 0:
                logger.info(
                    f"  Processing batch {batch_num:,}/{total_batches:,} ({batch_num / total_batches * 100:.1f}%)"
                )

            batch_df = self.df.iloc[start : start + batch_size]

            # Validate batch
            validation_result = self._validate_batch(batch_df, batch_size)
            if validation_result is not None:
                counters[validation_result] += 1
                continue

            # Get batch info
            csif_values = batch_df["CSIF"].values
            LAT = str(batch_df["latitude"].values[0])
            DF_DOY_S = batch_df["doy"].values[0]

            # Get config for this latitude
            if LAT not in self.phenology_config[self.config.PFT]:
                counters["skipped_no_config"] += 1
                continue

            lat_config = self.phenology_config[self.config.PFT][LAT]
            SOS_S = lat_config["sos"]["start"]
            SOS_E = lat_config["sos"]["end"]
            EOS_S = lat_config["eos"]["start"]
            EOS_E = lat_config["eos"]["end"]

            # ================================================================
            # SOS DETECTION
            # ================================================================
            sos_result = self._check_sos(
                csif_values, DF_DOY_S, SOS_S, SOS_E, batch_df.index[0]
            )

            if sos_result["detected"]:
                SOS_indices.append(batch_df.index[0])
                sos_starts.append(DF_DOY_S)
                sos_csif_means.append(sos_result["csif_mean"])
                sos_csif_increases.append(sos_result["csif_increase"])
                counters["sos_count"] += 1
            elif sos_result["skip_reason"]:
                counters[sos_result["skip_reason"]] += 1

            # ================================================================
            # EOS DETECTION
            # ================================================================
            eos_result = self._check_eos(
                csif_values, DF_DOY_S, EOS_S, EOS_E, batch_df.index[0]
            )

            if eos_result["detected"]:
                EOS_indices.append(batch_df.index[0])
                eos_starts.append(DF_DOY_S)
                eos_csif_means.append(eos_result["csif_mean"])
                eos_csif_decreases.append(eos_result["csif_decrease"])
                counters["eos_count"] += 1
            elif eos_result["skip_reason"]:
                counters[eos_result["skip_reason"]] += 1

        logger.info(f"  Processing batch {total_batches:,}/{total_batches:,} (100.0%)")

        # Compile results
        results = DetectionResults(
            SOS_indices=SOS_indices,
            EOS_indices=EOS_indices,
            sos_stats={
                "doy": sos_starts,
                "csif_mean": sos_csif_means,
                "csif_increase": sos_csif_increases,
            },
            eos_stats={
                "doy": eos_starts,
                "csif_mean": eos_csif_means,
                "csif_decrease": eos_csif_decreases,
            },
            skipped_stats=counters,
        )

        self._print_results(results)

        return results

    def _validate_batch(self, batch_df: pd.DataFrame, batch_size: int) -> Optional[str]:
        """Validate batch for basic requirements. Returns skip reason or None."""

        # Check completeness
        if len(batch_df) < batch_size:
            return "skipped_incomplete"

        # Check for NaNs
        if batch_df["CSIF"].isna().any():
            return "skipped_nan"

        # Check time continuity
        t = pd.to_datetime(batch_df["time"]).to_numpy(dtype="datetime64[D]")
        daydiff = (t[1:] - t[:-1]).astype("timedelta64[D]").astype(np.int64)

        if batch_df["location"].nunique() > 1 or not np.all(daydiff == 1):
            return "skipped_multi_location"

        return None

    def _check_sos(
        self,
        csif_values: np.ndarray,
        doy_start: int,
        sos_start: int,
        sos_end: int,
        index: int,
    ) -> Dict:
        """Check if batch qualifies as SOS"""

        result = {"detected": False, "skip_reason": None}

        # Check DOY window
        in_window = doy_start >= sos_start and (
            (doy_start < sos_end)
            or ((doy_start < sos_end + 365) and (sos_start > sos_end))
        )

        if not in_window:
            return result

        # Check monotonic increase
        if not all(x < y for x, y in zip(csif_values, csif_values[1:])):
            result["skip_reason"] = "skipped_sos_monotonic"
            return result

        # NEW: Check mean CSIF threshold
        mean_csif = np.mean(csif_values)
        if mean_csif < self.config.csif_threshold_sos:
            result["skip_reason"] = "skipped_sos_csif_threshold"
            return result

        # NEW: Check signal strength
        csif_increase = csif_values[-1] - csif_values[0]
        if csif_increase < self.config.min_csif_increase_sos:
            result["skip_reason"] = "skipped_sos_weak_signal"
            return result

        # All checks passed
        result["detected"] = True
        result["csif_mean"] = mean_csif
        result["csif_increase"] = csif_increase

        return result

    def _check_eos(
        self,
        csif_values: np.ndarray,
        doy_start: int,
        eos_start: int,
        eos_end: int,
        index: int,
    ) -> Dict:
        """Check if batch qualifies as EOS"""

        result = {"detected": False, "skip_reason": None}

        # Check DOY window
        in_window = doy_start >= eos_start and (
            doy_start < eos_end
            or ((doy_start < eos_end + 365) and (eos_start > eos_end))
        )

        if not in_window:
            return result

        # Check monotonic decrease
        if not all(x > y for x, y in zip(csif_values, csif_values[1:])):
            result["skip_reason"] = "skipped_eos_monotonic"
            return result

        # NEW: Check mean CSIF threshold (should still be positive)
        mean_csif = np.mean(csif_values)
        if mean_csif < self.config.csif_threshold_eos:
            result["skip_reason"] = "skipped_eos_csif_threshold"
            return result

        # NEW: Check signal strength (decrease should be significant)
        csif_decrease = csif_values[-1] - csif_values[0]
        if csif_decrease > self.config.max_csif_decrease_eos:
            result["skip_reason"] = "skipped_eos_weak_signal"
            return result

        # All checks passed
        result["detected"] = True
        result["csif_mean"] = mean_csif
        result["csif_decrease"] = csif_decrease

        return result

    def _print_results(self, results: DetectionResults):
        """Print detection results"""
        logger.info("\n" + "=" * 80)
        logger.info("DETECTION RESULTS")
        logger.info("=" * 80)

        logger.info("\nBatch Statistics:")
        logger.info(
            f"  Skipped (incomplete): {results.skipped_stats['skipped_incomplete']:,}"
        )
        logger.info(
            f"  Skipped (multi-location): {results.skipped_stats['skipped_multi_location']:,}"
        )
        logger.info(f"  Skipped (NaNs): {results.skipped_stats['skipped_nan']:,}")
        logger.info(
            f"  Skipped (no config): {results.skipped_stats['skipped_no_config']:,}"
        )

        logger.info("\nSOS Detection:")
        logger.info(f"  ✓ Detected: {results.skipped_stats['sos_count']:,}")
        logger.info(
            f"  ✗ Skipped (not monotonic): {results.skipped_stats['skipped_sos_monotonic']:,}"
        )
        logger.info(
            f"  ✗ Skipped (CSIF threshold): {results.skipped_stats['skipped_sos_csif_threshold']:,}"
        )
        logger.info(
            f"  ✗ Skipped (weak signal): {results.skipped_stats['skipped_sos_weak_signal']:,}"
        )

        logger.info("\nEOS Detection:")
        logger.info(f"  ✓ Detected: {results.skipped_stats['eos_count']:,}")
        logger.info(
            f"  ✗ Skipped (not monotonic): {results.skipped_stats['skipped_eos_monotonic']:,}"
        )
        logger.info(
            f"  ✗ Skipped (CSIF threshold): {results.skipped_stats['skipped_eos_csif_threshold']:,}"
        )
        logger.info(
            f"  ✗ Skipped (weak signal): {results.skipped_stats['skipped_eos_weak_signal']:,}"
        )

        if results.sos_stats["doy"]:
            logger.info("\nSOS Characteristics:")
            logger.info(
                f"  DOY - min: {min(results.sos_stats['doy']):.0f}, "
                f"max: {max(results.sos_stats['doy']):.0f}, "
                f"mean: {np.mean(results.sos_stats['doy']):.1f}"
            )
            logger.info(
                f"  CSIF - min: {min(results.sos_stats['csif_mean']):.3f}, "
                f"max: {max(results.sos_stats['csif_mean']):.3f}, "
                f"mean: {np.mean(results.sos_stats['csif_mean']):.3f}"
            )
            logger.info(
                f"  Increase - min: {min(results.sos_stats['csif_increase']):.3f}, "
                f"max: {max(results.sos_stats['csif_increase']):.3f}, "
                f"mean: {np.mean(results.sos_stats['csif_increase']):.3f}"
            )

        if results.eos_stats["doy"]:
            logger.info("\nEOS Characteristics:")
            logger.info(
                f"  DOY - min: {min(results.eos_stats['doy']):.0f}, "
                f"max: {max(results.eos_stats['doy']):.0f}, "
                f"mean: {np.mean(results.eos_stats['doy']):.1f}"
            )
            logger.info(
                f"  CSIF - min: {min(results.eos_stats['csif_mean']):.3f}, "
                f"max: {max(results.eos_stats['csif_mean']):.3f}, "
                f"mean: {np.mean(results.eos_stats['csif_mean']):.3f}"
            )


# ============================================================================
# VALIDATION
# ============================================================================


class Validator:
    """Validate detection results"""

    def __init__(self, config: Config, df: pd.DataFrame, results: DetectionResults):
        self.config = config
        self.df = df
        self.results = results

    def validate(self) -> Dict:
        """Run all validation checks"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION: Checking Phenological Consistency")
        logger.info("=" * 80)

        # Get SOS sample statistics
        sos_samples = self._extract_sample_stats(self.results.SOS_indices)

        # Check if we have valid samples
        if not sos_samples:
            logger.error("No valid SOS samples for validation!")
            logger.error("Check that indices match dataframe structure.")
            return {
                "csif_diff": 0.0,
                "max_jump": 0.0,
                "below_50_count": 0,
                "above_50_count": 0,
            }

        sos_df = pd.DataFrame(sos_samples)

        # Split by 50N boundary
        below_50 = sos_df[sos_df["latitude"] < 50]
        above_50 = sos_df[sos_df["latitude"] >= 50]

        # Print statistics
        logger.info("\n<50°N Region:")
        logger.info(f"  Samples: {len(below_50):,}")
        if len(below_50) > 0:
            logger.info(f"  Mean CSIF: {below_50['csif_mean'].mean():.4f}")
            logger.info(f"  Std CSIF: {below_50['csif_mean'].std():.4f}")
            logger.info(f"  Mean DOY: {below_50['doy_start'].mean():.1f}")
            logger.info(
                f"  Negative CSIF samples: {(below_50['csif_mean'] < 0).sum()} "
                f"({(below_50['csif_mean'] < 0).sum() / len(below_50) * 100:.1f}%)"
            )

        logger.info("\n≥50°N Region:")
        logger.info(f"  Samples: {len(above_50):,}")
        if len(above_50) > 0:
            logger.info(f"  Mean CSIF: {above_50['csif_mean'].mean():.4f}")
            logger.info(f"  Std CSIF: {above_50['csif_mean'].std():.4f}")
            logger.info(f"  Mean DOY: {above_50['doy_start'].mean():.1f}")
            logger.info(
                f"  Negative CSIF samples: {(above_50['csif_mean'] < 0).sum()} "
                f"({(above_50['csif_mean'] < 0).sum() / len(above_50) * 100:.1f}%)"
            )

        # Calculate CSIF difference
        csif_diff = 0.0
        if len(below_50) > 0 and len(above_50) > 0:
            csif_diff = abs(below_50["csif_mean"].mean() - above_50["csif_mean"].mean())
            logger.info(f"\nCSIF Difference: {csif_diff:.4f}")

            if csif_diff < 0.1:
                logger.info("✓ EXCELLENT! Similar phenological stage across latitudes")
            elif csif_diff < 0.2:
                logger.info("✓ GOOD! Much improved from original")
            else:
                logger.info("⚠️  May need stricter thresholds")

        # Check gradient smoothness
        max_jump = self._check_gradient(sos_df)

        # Create visualization if requested
        if self.config.create_plots:
            self._create_validation_plots(sos_df, csif_diff, max_jump)

        return {
            "csif_diff": csif_diff,
            "max_jump": max_jump,
            "below_50_count": len(below_50),
            "above_50_count": len(above_50),
        }

    def _extract_sample_stats(self, indices: List[int]) -> List[Dict]:
        """Extract statistics for samples at given indices"""
        samples = []
        skipped = 0

        for idx in indices:
            try:
                # Convert label index to position index
                if idx not in self.df.index:
                    skipped += 1
                    continue

                # Get position of this index in the dataframe
                pos = self.df.index.get_loc(idx)

                # Get the batch using position-based indexing
                batch = self.df.iloc[pos : pos + self.config.forecast_window]

                # Skip if batch is empty or too small
                if len(batch) == 0 or len(batch) < self.config.forecast_window:
                    skipped += 1
                    continue

                samples.append(
                    {
                        "latitude": batch["latitude"].values[0],
                        "longitude": batch["longitude"].values[0],
                        "csif_mean": batch["CSIF"].mean(),
                        "csif_min": batch["CSIF"].min(),
                        "csif_max": batch["CSIF"].max(),
                        "doy_start": batch["doy"].values[0],
                        "doy_end": batch["doy"].values[-1],
                    }
                )
            except Exception as e:
                skipped += 1
                logger.debug(f"Skipped index {idx}: {e}")
                continue

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped}/{len(indices)} samples during validation"
            )

        return samples

    def _check_gradient(self, sos_df: pd.DataFrame) -> float:
        """Check for smooth gradient across latitudes"""
        logger.info("\nGradient Check:")

        lat_bins = np.arange(self.config.lat_min, self.config.lat_max + 0.5, 1.0)
        csif_by_lat = sos_df.groupby(
            pd.cut(sos_df["latitude"], lat_bins, observed=False)
        )["csif_mean"].mean()

        jumps = np.abs(np.diff(csif_by_lat.dropna()))
        max_jump = jumps.max() if len(jumps) > 0 else 0.0
        mean_jump = jumps.mean() if len(jumps) > 0 else 0.0

        logger.info(f"  Maximum jump between 1° bins: {max_jump:.4f}")
        logger.info(f"  Average change: {mean_jump:.4f}")

        if max_jump < 0.15:
            logger.info("  ✓ Smooth gradient!")
        else:
            logger.info("  ⚠️  Some sharp transitions")

        return max_jump

    def _create_validation_plots(
        self, sos_df: pd.DataFrame, csif_diff: float, max_jump: float
    ):
        """Create validation visualizations"""
        logger.info("\nCreating validation plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: CSIF by latitude
        ax = axes[0, 0]
        lat_bins = np.arange(self.config.lat_min, self.config.lat_max + 0.5, 1.0)
        csif_by_lat = sos_df.groupby(
            pd.cut(sos_df["latitude"], lat_bins, observed=False)
        )["csif_mean"].mean()
        lat_centers = [
            (lat_bins[i] + lat_bins[i + 1]) / 2 for i in range(len(lat_bins) - 1)
        ]

        ax.plot(lat_centers, csif_by_lat.values, "go-", linewidth=2, markersize=8)
        ax.axvline(
            x=50, color="black", linestyle="--", linewidth=2, alpha=0.5, label="50°N"
        )
        ax.axhline(
            y=self.config.csif_threshold_sos,
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Threshold ({self.config.csif_threshold_sos})",
        )
        ax.set_xlabel("Latitude (°N)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean CSIF", fontsize=12, fontweight="bold")
        ax.set_title(
            f"CSIF Gradient (diff={csif_diff:.4f})",
            fontsize=13,
            fontweight="bold",
            color="green",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: DOY by latitude
        ax = axes[0, 1]
        doy_by_lat = sos_df.groupby(
            pd.cut(sos_df["latitude"], lat_bins, observed=False)
        )["doy_start"].mean()
        ax.plot(lat_centers, doy_by_lat.values, "bo-", linewidth=2, markersize=8)
        ax.axvline(x=50, color="black", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_xlabel("Latitude (°N)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean DOY", fontsize=12, fontweight="bold")
        ax.set_title("SOS Timing by Latitude", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Plot 3: Sample distribution
        ax = axes[0, 2]
        ax.hist(
            sos_df[sos_df["latitude"] < 50]["csif_mean"],
            bins=30,
            alpha=0.6,
            label="<50°N",
            color="blue",
            density=True,
        )
        ax.hist(
            sos_df[sos_df["latitude"] >= 50]["csif_mean"],
            bins=30,
            alpha=0.6,
            label="≥50°N",
            color="red",
            density=True,
        )
        ax.axvline(x=0, color="green", linestyle="--", linewidth=2, label="CSIF=0")
        ax.set_xlabel("Mean CSIF", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax.set_title("CSIF Distribution", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Geographic distribution
        ax = axes[1, 0]
        scatter = ax.scatter(
            sos_df["longitude"],
            sos_df["latitude"],
            c=sos_df["csif_mean"],
            cmap="RdYlGn",
            s=20,
            alpha=0.6,
            vmin=-0.5,
            vmax=1.0,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.axhline(y=50, color="black", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
        ax.set_title("Geographic Distribution", fontsize=13, fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="CSIF")
        ax.grid(True, alpha=0.3)

        # Plot 5: DOY vs CSIF scatter
        ax = axes[1, 1]
        ax.scatter(sos_df["doy_start"], sos_df["csif_mean"], alpha=0.3, s=10, c="green")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_xlabel("DOY Start", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean CSIF", fontsize=12, fontweight="bold")
        ax.set_title("DOY vs CSIF", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Plot 6: Summary stats
        ax = axes[1, 2]
        ax.axis("off")

        summary_text = f"""
VALIDATION SUMMARY
{"=" * 50}

Dataset: {self.config.PFT}

Samples:
  Total SOS: {len(sos_df):,}
  <50°N: {len(sos_df[sos_df["latitude"] < 50]):,}
  ≥50°N: {len(sos_df[sos_df["latitude"] >= 50]):,}

Quality Metrics:
  CSIF difference: {csif_diff:.4f}
  Max gradient jump: {max_jump:.4f}
  Negative samples: {(sos_df["csif_mean"] < 0).sum()}

Thresholds Used:
  CSIF threshold: {self.config.csif_threshold_sos}
  Min increase: {self.config.min_csif_increase_sos}

Status: {"✓ PASSED" if csif_diff < 0.1 else "⚠ REVIEW"}
"""

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()

        output_file = f"{self.config.output_dir}{self.config.PFT}_validation.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"✓ Saved validation plots: {output_file}")
        plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline(config: Config):
    """Run the complete SOS/EOS detection pipeline"""

    logger.info("\n" + "=" * 80)
    logger.info(f"SOS/EOS DETECTION PIPELINE - {config.PFT}")
    logger.info("=" * 80)

    # 1. Load data
    data_loader = DataLoader(config)
    df = data_loader.load_data()
    phenology_config = data_loader.load_config()

    # 2. Detect phenology
    detector = PhenologyDetector(config, df, phenology_config)
    results = detector.detect()

    # 3. Validate
    validator = Validator(config, df, results)
    validation_metrics = validator.validate()

    # 4. Save results
    output_dict = {"SOS": results.SOS_indices, "EOS": results.EOS_indices}

    output_file = config.get_output_file(corrected=True)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(output_dict, f)

    logger.info(f"\n✓ Saved indices: {output_file}")

    # Also save full results for analysis
    full_results = {
        "SOS": results.SOS_indices,
        "EOS": results.EOS_indices,
        "sos_stats": results.sos_stats,
        "eos_stats": results.eos_stats,
        "skipped_stats": results.skipped_stats,
        "validation": validation_metrics,
    }

    full_output_file = output_file.replace(".pkl", "_FULL.pkl")
    with open(full_output_file, "wb") as f:
        pickle.dump(full_results, f)

    logger.info(f"✓ Saved full results: {full_output_file}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nDataset: {config.PFT}")
    logger.info(f"SOS samples: {len(results.SOS_indices):,}")
    logger.info(f"EOS samples: {len(results.EOS_indices):,}")
    logger.info(f"CSIF difference: {validation_metrics['csif_diff']:.4f}")

    if validation_metrics["csif_diff"] < 0.1:
        logger.info("\n🎉 SUCCESS! Phenological consistency achieved")
    elif validation_metrics["csif_diff"] < 0.2:
        logger.info("\n✓ IMPROVED! But could be better - try adjusting thresholds")
    else:
        logger.info("\n⚠️  Still issues - increase thresholds")

    logger.info("\n" + "=" * 80)

    return results, validation_metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run for BDT_50_90
    logger.info("\n\n" + "#" * 80)
    logger.info("# PROCESSING BDT_50_90")
    logger.info("#" * 80)

    config_50_90 = Config(PFT="BDT_50_90")
    results_50_90, metrics_50_90 = run_pipeline(config_50_90)

    # Run for BDT_50_20
    logger.info("\n\n" + "#" * 80)
    logger.info("# PROCESSING BDT_50_20")
    logger.info("#" * 80)

    config_50_20 = Config(PFT="BDT_50_20")
    results_50_20, metrics_50_20 = run_pipeline(config_50_20)

    # Final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info("\nBDT_50_90:")
    logger.info(f"  SOS samples: {len(results_50_90.SOS_indices):,}")
    logger.info(f"  CSIF diff: {metrics_50_90['csif_diff']:.4f}")
    logger.info("\nBDT_50_20:")
    logger.info(f"  SOS samples: {len(results_50_20.SOS_indices):,}")
    logger.info(f"  CSIF diff: {metrics_50_20['csif_diff']:.4f}")

    logger.info("\n✓ All processing complete!")
    logger.info("=" * 80)
