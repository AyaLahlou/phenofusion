"""
REFACTORED FEATURE IMPORTANCE EXTRACTION
=========================================

Extract feature importance (attention weights) for corrected SOS/EOS samples.

Key improvements:
- Works for both BDT_50_20 and BDT_50_90 datasets
- Modular, clean code structure
- Comprehensive error handling
- Progress tracking
- Automatic validation

Author: Refactored from original notebook
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import pickle
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
    """Configuration for feature importance extraction"""

    # Dataset selection
    PFT: str = "BDT_50_90"  # Options: "BDT_50_20", "BDT_50_90"

    # Paths
    pred_dir: str = "/burg-archive/home/al4385/phenofusion/src/phenofusion/predictions/"
    indices_dir: str = "/burg-archive/home/al4385/phenofusion/src/phenofusion/debugging/sos_eos_indices_pickle/"
    output_dir: str = (
        "/burg-archive/home/al4385/phenofusion/src/phenofusion/feature_importance/"
    )

    # Parameters
    forecast_window: int = 30
    max_window_start: int = 336  # Maximum start index for attention window

    # Feature names
    feature_names: List[str] = None

    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = ["tmin", "tmax", "rad", "precip", "photo", "sm"]

    def get_pred_file(self) -> str:
        """Get predictions file path"""
        return f"{self.pred_dir}{self.PFT}_predictions.pkl"

    def get_indices_file(self, corrected: bool = True) -> str:
        """Get indices file path"""
        suffix = "_CORRECTED" if corrected else ""
        return f"{self.indices_dir}subset_{self.PFT}_sos_eos_dict{suffix}.pkl"

    def get_output_file(self, phase: str) -> str:
        """Get output file path for SOS or EOS"""
        return f"{self.output_dir}{self.PFT}_subset_{phase}.csv"


# ============================================================================
# DATA LOADING
# ============================================================================


class DataLoader:
    """Load predictions and indices"""

    def __init__(self, config: Config):
        self.config = config

    def load_predictions(self) -> Dict:
        """Load model predictions"""
        pred_file = self.config.get_pred_file()
        logger.info(f"Loading predictions from {pred_file}")

        with open(pred_file, "rb") as f:
            preds = pickle.load(f)

        logger.info("✓ Predictions loaded")
        logger.info(f"  Keys: {list(preds.keys())}")
        logger.info(f"  Attention scores shape: {preds['attention_scores'].shape}")
        logger.info(
            f"  Historical weights shape: {preds['historical_selection_weights'].shape}"
        )

        return preds

    def load_indices(self, corrected: bool = True) -> Dict:
        """Load SOS/EOS indices"""
        indices_file = self.config.get_indices_file(corrected=corrected)
        logger.info(f"Loading indices from {indices_file}")

        with open(indices_file, "rb") as f:
            indices = pickle.load(f)

        logger.info("✓ Indices loaded")
        logger.info(f"  SOS samples: {len(indices['SOS']):,}")
        logger.info(f"  EOS samples: {len(indices['EOS']):,}")

        return indices


# ============================================================================
# FEATURE IMPORTANCE EXTRACTION
# ============================================================================


class FeatureImportanceExtractor:
    """Extract feature importance from attention weights"""

    def __init__(self, config: Config, preds: Dict):
        self.config = config
        self.preds = preds

    def extract_for_phase(self, indices: List[int], phase: str) -> pd.DataFrame:
        """
        Extract feature importance for all samples in a phenological phase

        Args:
            indices: List of dataframe indices (not prediction indices!)
            phase: "SOS" or "EOS"

        Returns:
            DataFrame with feature importance for each sample
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EXTRACTING FEATURE IMPORTANCE: {phase}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Processing {len(indices):,} samples...")

        # Convert dataframe indices to prediction indices
        pred_indices = [int(i / self.config.forecast_window) for i in indices]

        logger.info(f"Converted to {len(pred_indices):,} prediction indices")

        driver_data = []
        skipped_window = 0
        errors = 0

        # Progress tracking
        report_interval = max(100, len(pred_indices) // 20)

        for count, index in enumerate(pred_indices):
            if count % report_interval == 0:
                logger.info(
                    f"  Processing sample {count:,}/{len(pred_indices):,} ({count / len(pred_indices) * 100:.1f}%)"
                )

            try:
                # Find best attention window
                window_start = self.find_max_attention_window(
                    index, self.preds, self.config.forecast_window
                )

                # Skip if window is too far
                if window_start > self.config.max_window_start:
                    skipped_window += 1
                    continue

                # Extract weights
                weights = {}
                hist_weights = self.preds["historical_selection_weights"][index]

                # Get feature-specific weights
                for i, var in enumerate(
                    self.config.feature_names[:-1]
                ):  # All except sm
                    var_weights = hist_weights[
                        i, window_start : window_start + self.config.forecast_window
                    ]
                    weights[f"hist_{var}"] = np.mean(var_weights)

                # Get soil moisture weights (last feature)
                sm_weights = hist_weights[
                    -1, window_start : window_start + self.config.forecast_window
                ]
                weights["hist_sm"] = np.mean(sm_weights)

                # Get location info (from original dataframe index)
                df_index = indices[count]
                weights["original_index"] = df_index
                weights["prediction_index"] = index

                driver_data.append(weights)

            except Exception as e:
                errors += 1
                if errors <= 5:  # Only log first few errors
                    logger.warning(f"Error processing index {index}: {e}")

        logger.info(
            f"  Processing sample {len(pred_indices):,}/{len(pred_indices):,} (100.0%)"
        )

        logger.info("\nExtraction Summary:")
        logger.info(f"  ✓ Successfully processed: {len(driver_data):,}")
        logger.info(f"  ✗ Skipped (window too far): {skipped_window:,}")
        logger.info(f"  ✗ Errors: {errors:,}")

        # Create DataFrame
        driver_df = pd.DataFrame(driver_data)

        # Normalize weights to sum to 1
        feature_cols = [f"hist_{var}" for var in self.config.feature_names]
        driver_df[feature_cols] = driver_df[feature_cols].div(
            driver_df[feature_cols].sum(axis=1), axis=0
        )

        logger.info("\nFeature Importance Statistics:")
        for col in feature_cols:
            logger.info(
                f"  {col}: mean={driver_df[col].mean():.4f}, "
                f"std={driver_df[col].std():.4f}"
            )

        return driver_df

    def find_max_attention_window(
        self, index: int, preds: Dict, forecast_window: int
    ) -> int:
        """Find the time window with maximum attention - fixed signature"""
        att_array = np.mean(preds["attention_scores"][index], axis=0)

        max_sum = -np.inf
        best_start_index = 0

        max_start = min(336, len(att_array) - forecast_window)

        for i in range(max_start):
            current_sum = np.sum(att_array[i : i + forecast_window])

            if current_sum > max_sum:
                max_sum = current_sum
                best_start_index = i

        return best_start_index


# ============================================================================
# VALIDATION
# ============================================================================


class FeatureImportanceValidator:
    """Validate extracted feature importance"""

    def __init__(
        self,
        config: Config,
        sos_df: pd.DataFrame,
        eos_df: Optional[pd.DataFrame] = None,
    ):
        self.config = config
        self.sos_df = sos_df
        self.eos_df = eos_df

    def validate(self):
        """Run validation checks"""
        logger.info(f"\n{'=' * 80}")
        logger.info("VALIDATION: Feature Importance")
        logger.info(f"{'=' * 80}")

        # Check SOS
        self._validate_phase(self.sos_df, "SOS")

        # Check EOS if available
        if self.eos_df is not None and len(self.eos_df) > 0:
            self._validate_phase(self.eos_df, "EOS")

    def _validate_phase(self, df: pd.DataFrame, phase: str):
        """Validate feature importance for one phase"""
        logger.info(f"\n{phase} Validation:")

        feature_cols = [f"hist_{var}" for var in self.config.feature_names]

        # Check sums to 1
        row_sums = df[feature_cols].sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            logger.warning("  ⚠️  Not all rows sum to 1.0!")
            logger.warning(
                f"     Min sum: {row_sums.min():.6f}, Max sum: {row_sums.max():.6f}"
            )
        else:
            logger.info("  ✓ All importance weights sum to 1.0")

        # Check for NaNs
        nan_count = df[feature_cols].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"  ⚠️  Found {nan_count} NaN values!")
        else:
            logger.info("  ✓ No NaN values")

        # Check for negative values
        neg_count = (df[feature_cols] < 0).sum().sum()
        if neg_count > 0:
            logger.warning(f"  ⚠️  Found {neg_count} negative values!")
        else:
            logger.info("  ✓ No negative values")

        # Print dominant features
        logger.info("\n  Top Features:")
        mean_importance = df[feature_cols].mean().sort_values(ascending=False)
        for feat, imp in mean_importance.items():
            logger.info(f"    {feat}: {imp:.4f}")


# ============================================================================
# MERGER WITH ORIGINAL DATA
# ============================================================================


class DataMerger:
    """Merge feature importance with original dataframe"""

    def __init__(self, config: Config):
        self.config = config

    def merge_with_original(
        self, importance_df: pd.DataFrame, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge feature importance with original data

        Args:
            importance_df: Feature importance dataframe
            original_df: Original analysis dataframe

        Returns:
            Merged dataframe
        """
        logger.info("\nMerging with original data...")

        # Get original data for the indices
        original_subset = original_df.loc[importance_df["original_index"]]

        # Reset index to allow concatenation
        original_subset = original_subset.reset_index(drop=True)
        importance_df = importance_df.reset_index(drop=True)

        # Concatenate
        merged_df = pd.concat([original_subset, importance_df], axis=1)

        logger.info(f"✓ Merged dataframe shape: {merged_df.shape}")

        return merged_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline(config: Config, use_corrected: bool = True):
    """
    Run the complete feature importance extraction pipeline

    Args:
        config: Configuration object
        use_corrected: Whether to use corrected indices
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FEATURE IMPORTANCE PIPELINE - {config.PFT}")
    logger.info(f"Using {'CORRECTED' if use_corrected else 'ORIGINAL'} indices")
    logger.info(f"{'=' * 80}")

    # 1. Load data
    data_loader = DataLoader(config)
    preds = data_loader.load_predictions()
    indices = data_loader.load_indices(corrected=use_corrected)

    # 2. Extract feature importance
    extractor = FeatureImportanceExtractor(config, preds)

    sos_importance = extractor.extract_for_phase(indices["SOS"], "SOS")
    eos_importance = extractor.extract_for_phase(indices["EOS"], "EOS")

    # 3. Validate
    validator = FeatureImportanceValidator(config, sos_importance, eos_importance)
    validator.validate()

    # 4. Save results
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Save SOS
    sos_output = config.get_output_file("SOS")
    sos_importance.to_csv(sos_output, index=False)
    logger.info(f"\n✓ Saved SOS importance: {sos_output}")

    # Save EOS
    eos_output = config.get_output_file("EOS")
    eos_importance.to_csv(eos_output, index=False)
    logger.info(f"✓ Saved EOS importance: {eos_output}")

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"\nDataset: {config.PFT}")
    logger.info(f"SOS samples: {len(sos_importance):,}")
    logger.info(f"EOS samples: {len(eos_importance):,}")
    logger.info("\nOutput files:")
    logger.info(f"  {sos_output}")
    logger.info(f"  {eos_output}")
    logger.info(f"{'=' * 80}\n")

    return sos_importance, eos_importance


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Process BDT_50_90
    logger.info("\n\n" + "#" * 80)
    logger.info("# PROCESSING BDT_50_90")
    logger.info("#" * 80)

    config_50_90 = Config(PFT="BDT_50_90")
    sos_50_90, eos_50_90 = run_pipeline(config_50_90, use_corrected=True)

    # Process BDT_50_20
    logger.info("\n\n" + "#" * 80)
    logger.info("# PROCESSING BDT_50_20")
    logger.info("#" * 80)

    config_50_20 = Config(PFT="BDT_50_20")
    sos_50_20, eos_50_20 = run_pipeline(config_50_20, use_corrected=True)

    # Final summary
    logger.info("\n\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info("\nBDT_50_90:")
    logger.info(f"  SOS samples: {len(sos_50_90):,}")
    logger.info(f"  EOS samples: {len(eos_50_90):,}")
    logger.info("\nBDT_50_20:")
    logger.info(f"  SOS samples: {len(sos_50_20):,}")
    logger.info(f"  EOS samples: {len(eos_50_20):,}")

    logger.info("\n✓ All processing complete!")
    logger.info("=" * 80)
