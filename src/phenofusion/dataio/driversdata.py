"""
Driver Data Analysis Module

This module provides functionality for analyzing attention weights and driver data
from phenology prediction models. It includes utilities for data processing,
visualization, and generating driver attention maps.

The `main()` function serves as the entry point when running this script directly.

Author: Aya Lahlou
Date: October 29th, 2025
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from scipy.stats import linregress
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhenologyConfig:
    """Configuration class for phenology analysis parameters."""

    # Default detection thresholds
    DEFAULT_MIN_DIFF = 0.20
    DEFAULT_MIN_SLOPE = 0.002

    # PFT-specific thresholds
    PFT_THRESHOLDS = {
        "BET": {"min_diff": 0.05, "min_slope": 0.001},
        "SHR": {"min_diff": 0.05, "min_slope": 0.001},
    }

    # BET latitude bands and corresponding DOY ranges
    BET_PHENOLOGY_BANDS = {
        (30.5, 35): {"sos": (40, 140), "eos": (210, 310)},
        (27, 30.5): {"sos": (50, 150), "eos": (200, 300)},
        (25.75, 27): {"sos": (25, 150), "eos": (210, 310)},
        (20, 25.5): {"sos": (40, 170), "eos": (220, 315)},
        (13, 19.75): {"sos": (25, 160), "eos": (230, 315)},
        (10, 12.75): {"sos": (25, 160), "eos": (240, 325)},
        (9, 9.75): {"sos": (10, 110), "eos": (250, 315)},
        (7, 9): {"sos": (10, 90), "eos": (260, 325)},
        (4, 7): {"sos": (25, 75), "eos": (260, 325)},
        (2, 4): {"sos": (20, 70), "eos": (275, 340)},
        (-2, 2): {"sos": [(20, 70), (190, 250)], "eos": [(90, 150), (275, 340)]},
        (-6.5, -2): {"sos": (190, 250), "eos": (80, 150)},
        (-9.5, -6.5): {"sos": (180, 250), "eos": (60, 150)},
        (-12, -9.5): {"sos": (190, 275), "eos": (50, 120)},
        (-16.5, -12): {"sos": (200, 275), "eos": (50, 120)},
        (-18, -16.5): {"sos": (220, 300), "eos": (40, 120)},
        (-20.75, -18): {"sos": (240, 320), "eos": (40, 120)},
        (-22, -20.75): {"sos": (200, 275), "eos": (40, 120)},
        (-22.75, -22): {"sos": (240, 320), "eos": (40, 120)},
        (-37.75, -22.75): {"sos": (200, 320), "eos": (30, 110)},
        (-41, -37.75): {"sos": (200, 300), "eos": (0, 100)},
        (-45.75, -41): {"sos": (200, 300), "eos": (20, 120)},
        (-90, -45.75): {"sos": (200, 300), "eos": (30, 120)},
    }


class DataProcessor:
    """Handles data processing and normalization operations."""

    @staticmethod
    def normalize_csif(group: pd.DataFrame) -> pd.DataFrame:
        """Normalize CSIF values within a group."""
        csif_min = group["CSIF"].min()
        csif_max = group["CSIF"].max()
        if csif_max == csif_min:
            group["CSIF_normalized"] = 0
        else:
            group["CSIF_normalized"] = (group["CSIF"] - csif_min) / (
                csif_max - csif_min
            )
        return group

    @staticmethod
    def compute_slope(group: pd.DataFrame) -> pd.Series:
        """Compute linear regression slope for CSIF time series."""
        x = range(len(group))
        y = group["CSIF"].values
        slope, _, _, _, _ = linregress(x, y)
        return pd.Series({"slope": slope})

    @staticmethod
    def get_analysis_df(data: Dict, preds: Dict, coord_path: str) -> pd.DataFrame:
        """
        Match original data to predictions and corresponding coordinates.

        Parameters
        ----------
        data : dict
            Original processed data dictionary
        preds : dict
            Predictions dictionary
        coord_path : str
            Path to dataframe matching location index to coordinates

        Returns
        -------
        pd.DataFrame
            DataFrame with groundtruth values, predictions and coordinates
        """
        logger.info("Loading coordinates and preparing analysis dataframe")

        coords = pd.read_parquet(coord_path).drop_duplicates()

        # Create main dataframe from test data
        df = pd.DataFrame(
            {
                "Index": data["data_sets"]["test"]["id"].flatten(),
                "Flattened_Values": data["data_sets"]["test"]["target"].flatten(),
            }
        )

        # Add predictions
        df["pred_05"] = preds["predicted_quantiles"][:, :, 1].flatten()

        # Parse index into location and time components
        df[["location_id", "time_id"]] = df["Index"].str.split("_", n=1, expand=True)
        df["location_id"] = df["location_id"].astype(int)
        df["time_id"] = pd.to_datetime(df["time_id"])

        # Sort and add time features
        df = df.sort_values(by=["location_id", "time_id"])
        df["doy"] = df["time_id"].dt.dayofyear
        df["year"] = df["time_id"].dt.year
        df["month"] = df["time_id"].dt.month
        df["day"] = df["time_id"].dt.day

        # Rename columns and merge with coordinates
        df = df.rename(
            columns={
                "Flattened_Values": "CSIF",
                "location_id": "location",
                "time_id": "time",
            }
        ).drop(columns=["Index"])

        df = pd.merge(coords, df, on="location", how="left")

        logger.info(f"Analysis dataframe created with {len(df)} rows")
        return df


class AttentionAnalyzer:
    """Handles attention weight analysis and driver mapping."""

    @staticmethod
    def max_attention_window(preds: Dict, index: int, forecast_window: int = 30) -> int:
        """
        Find the time window with maximum attention for a given prediction index.

        Parameters
        ----------
        preds : dict
            Predictions dictionary containing attention scores
        index : int
            Index of the prediction to analyze
        forecast_window : int
            Size of the forecast window

        Returns
        -------
        int
            Starting index of the maximum attention window
        """
        att_array = np.mean(preds["attention_scores"][index], axis=0)

        max_sum = -np.inf
        best_start_index = None

        for i in range(396 - (forecast_window) * 2):
            current_sum = np.sum(att_array[i : i + forecast_window])
            if current_sum > max_sum:
                max_sum = current_sum
                best_start_index = i

        return best_start_index

    @staticmethod
    def get_attention_weights_df(
        index_list: List[int], preds: Dict, data: Dict, forecast_window: int = 30
    ) -> pd.DataFrame:
        """
        Extract attention weights for environmental drivers.

        Parameters
        ----------
        index_list : list
            List of prediction indices to analyze
        preds : dict
            Predictions dictionary
        data : dict
            Original data dictionary
        forecast_window : int
            Forecast window size

        Returns
        -------
        pd.DataFrame
            DataFrame with location and driver attention weights
        """
        logger.info(f"Extracting attention weights for {len(index_list)} indices")

        df_attention_map = pd.DataFrame(
            columns=[
                "location",
                "legacy window start",
                "hist_tmin",
                "hist_tmax",
                "hist_rad",
                "hist_precip",
                "hist_photo",
                "hist_sm",
            ]
        )

        for index in index_list:
            window_start = AttentionAnalyzer.max_attention_window(
                preds, index, forecast_window
            )

            if window_start is not None and window_start <= 365 - forecast_window:
                # Extract median weights for each driver
                weights = {}
                driver_indices = {
                    "hist_tmin": 1,
                    "hist_tmax": 2,
                    "hist_rad": 3,
                    "hist_precip": 4,
                    "hist_photo": 5,
                    "hist_sm": 6,
                }

                for driver, driver_idx in driver_indices.items():
                    weights[driver] = np.median(
                        preds["historical_selection_weights"][
                            index,
                            window_start : window_start + forecast_window,
                            driver_idx,
                        ]
                    )

                location = int(data["data_sets"]["test"]["id"][index][0].split("_")[0])

                new_row = pd.DataFrame(
                    {
                        "location": [location],
                        "legacy window start": [window_start],
                        **{k: [v] for k, v in weights.items()},
                    }
                )
                df_attention_map = pd.concat(
                    [df_attention_map, new_row], ignore_index=True
                )

        logger.info(
            f"Extracted attention weights for {len(df_attention_map)} locations"
        )
        return df_attention_map


class PhenologyDetector:
    """Detects start and end of season events from time series data."""

    def __init__(self, config: PhenologyConfig):
        self.config = config

    def detect_slope_based(
        self,
        df: pd.DataFrame,
        test_mode: bool = False,
        batch_size: int = 30,
        pft: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Detect SOS and EOS using slope-based method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with CSIF time series
        batch_size : int
            Size of batches to process
        pft : str, optional
            Plant functional type for specific thresholds

        Returns
        -------
        tuple
            Lists of SOS and EOS indices
        """
        logger.info(f"Running slope-based detection for PFT: {pft}")

        # Get thresholds
        if pft and pft in self.config.PFT_THRESHOLDS:
            thresholds = self.config.PFT_THRESHOLDS[pft]
        else:
            thresholds = {
                "min_diff": self.config.DEFAULT_MIN_DIFF,
                "min_slope": self.config.DEFAULT_MIN_SLOPE,
            }

        sos_indices = []
        eos_indices = []

        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]

            x = range(len(batch_df))
            y = batch_df["CSIF"].values
            if test_mode:
                # In test mode, assume all batches are valid
                sos_indices.append(batch_df.index[0])
                eos_indices.append(batch_df.index[0])
            else:
                if abs(y[0] - y[-1]) > thresholds["min_diff"]:
                    slope, _, _, _, _ = linregress(x, y)

                    if slope >= thresholds["min_slope"]:
                        sos_indices.append(batch_df.index[0])
                    elif slope <= -thresholds["min_slope"]:
                        eos_indices.append(batch_df.index[0])

        logger.info(f"Found {len(sos_indices)} SOS and {len(eos_indices)} EOS events")
        return sos_indices, eos_indices

    def detect_bet_phenology(
        self,
        df: pd.DataFrame,
        test_mode: bool = False,
        batch_size: int = 30,
        buffer_days: int = 40,
    ) -> Tuple[List[int], List[int]]:
        """
        Detect BET phenology using latitude-DOY based method.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with CSIF time series
        batch_size : int
            Size of batches to process
        buffer_days : int
            Buffer days around expected DOY ranges

        Returns
        -------
        tuple
            Lists of SOS and EOS indices
        """
        logger.info("Running BET-specific phenology detection")

        sos_indices = []
        eos_indices = []
        total_batches = 0
        matched_batches = 0

        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]
            total_batches += 1

            lat = batch_df["latitude"].iloc[0]
            doy = batch_df["doy"].iloc[0]
            doy_2 = batch_df["doy"].iloc[-1]

            # Check against latitude bands for both SOS and EOS
            if test_mode:
                sos_match = True
                eos_match = True
            else:
                sos_match = self._check_bet_bands(lat, doy, doy_2, "sos", buffer_days)
                eos_match = self._check_bet_bands(lat, doy, doy_2, "eos", buffer_days)

            if sos_match:
                sos_indices.append(batch_df.index[0])
                matched_batches += 1
            elif eos_match:
                eos_indices.append(batch_df.index[0])
                matched_batches += 1

        logger.info(
            f"BET Detection - Total: {total_batches}, "
            f"Matched: {matched_batches} ({matched_batches/total_batches*100:.2f}%)"
        )
        logger.info(f"Found {len(sos_indices)} SOS and {len(eos_indices)} EOS events")

        return sos_indices, eos_indices

    def _check_bet_bands(
        self, lat: float, doy: int, doy_2: int, event_type: str, buffer_days: int
    ) -> bool:
        """Check if the DOY range matches expected latitude bands for given event type."""
        for (lat_min, lat_max), phenology in self.config.BET_PHENOLOGY_BANDS.items():
            if lat_min <= lat <= lat_max:
                # Handle bimodal phenology (tropical regions)
                if isinstance(phenology[event_type], list):
                    for event_range in phenology[event_type]:
                        if (
                            doy >= event_range[0] - buffer_days
                            and doy_2 <= event_range[1] + buffer_days
                        ):
                            return True
                else:
                    # Standard unimodal phenology
                    if (
                        doy >= phenology[event_type][0] - buffer_days
                        and doy_2 <= phenology[event_type][1] + buffer_days
                    ):
                        return True
                break
        return False


class SpatialProcessor:
    """Handles spatial data processing and imputation."""

    @staticmethod
    def impute_nearby(
        df: pd.DataFrame, lat_range: float = 0.5, lon_range: float = 0.5
    ) -> pd.DataFrame:
        """
        Impute missing values using nearby spatial locations.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with spatial coordinates
        lat_range : float
            Latitude range for nearby search
        lon_range : float
            Longitude range for nearby search

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values
        """
        logger.info("Imputing missing values using spatial neighbors")

        driver_cols = [
            "hist_tmin",
            "hist_tmax",
            "hist_rad",
            "hist_precip",
            "hist_photo",
            "hist_sm",
        ]

        # Identify rows with missing driver columns
        missing_mask = df[driver_cols].isnull().all(axis=1)
        non_missing_mask = ~missing_mask

        # For each missing row, find the nearest non-missing row within the spatial window and impute
        missing_rows = df[missing_mask].copy()
        for idx, missing_row in missing_rows.iterrows():
            lat, lon = missing_row["latitude"], missing_row["longitude"]
            # Find candidate rows within the spatial window
            candidates = df[
                (non_missing_mask)
                & (df["latitude"] >= lat - lat_range)
                & (df["latitude"] <= lat + lat_range)
                & (df["longitude"] >= lon - lon_range)
                & (df["longitude"] <= lon + lon_range)
            ]
            if not candidates.empty:
                # Use the first candidate (could use nearest if desired)
                candidate_row = candidates.iloc[0]
                df.loc[idx, driver_cols] = candidate_row[driver_cols].values

        return df


class AnalysisWorkflow:
    """Main workflow orchestrator for phenology driver analysis."""

    def __init__(self, config: Optional[PhenologyConfig] = None):
        self.config = config or PhenologyConfig()
        self.data_processor = DataProcessor()
        self.attention_analyzer = AttentionAnalyzer()
        self.phenology_detector = PhenologyDetector(self.config)
        self.spatial_processor = SpatialProcessor()

    def run_analysis(
        self,
        data_path: str,
        pred_path: str,
        coord_path: str,
        output_path: str,
        forecast_window: int,
        pft: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """
        Run the complete analysis workflow.

        Parameters
        ----------
        data_path : str
            Path to data pickle file
        pred_path : str
            Path to predictions pickle file
        coord_path : str
            Path to coordinates parquet file
        output_path : str
            Output path prefix
        forecast_window : int
            Forecast window size
        pft : str, optional
            Plant functional type
        """
        logger.info("Starting phenology driver analysis workflow")

        # Load data
        logger.info("Loading data files...")
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)
        with open(pred_path, "rb") as fp:
            preds = pickle.load(fp)

        # Prepare analysis dataframe
        df = self.data_processor.get_analysis_df(data, preds, coord_path)

        # Detect phenology events
        if pft == "BET":
            sos_indices, eos_indices = self.phenology_detector.detect_bet_phenology(
                df, test_mode=test_mode
            )
        else:
            sos_indices, eos_indices = self.phenology_detector.detect_slope_based(
                df, pft=pft, test_mode=test_mode
            )

        # Convert to prediction indices and filter
        sos_pred_indices = [int(i / 30) for i in sos_indices]
        eos_pred_indices = [int(i / 30) for i in eos_indices]

        max_pred_index = len(preds["attention_scores"]) - 1
        sos_pred_indices = [idx for idx in sos_pred_indices if idx <= max_pred_index]
        eos_pred_indices = [idx for idx in eos_pred_indices if idx <= max_pred_index]

        logger.info(
            f"Valid indices - SOS: {len(sos_pred_indices)}, EOS: {len(eos_pred_indices)}"
        )

        # Generate and save attention weight maps
        self._save_attention_maps(
            sos_pred_indices,
            eos_pred_indices,
            data,
            preds,
            coord_path,
            output_path,
            forecast_window,
        )

        logger.info("Analysis workflow completed successfully")

    def _save_attention_maps(
        self,
        sos_indices: List[int],
        eos_indices: List[int],
        data: Dict,
        preds: Dict,
        coord_path: str,
        output_path: str,
        forecast_window: int,
    ) -> None:
        """Save attention weight maps for SOS and EOS events."""
        coords = pd.read_parquet(coord_path).drop_duplicates()

        for indices, season in [(sos_indices, "SOS"), (eos_indices, "EOS")]:
            if indices:
                logger.info(f"Processing {season} attention weights")

                # Get attention weights
                df_attention = self.attention_analyzer.get_attention_weights_df(
                    indices, preds, data, forecast_window
                )

                # Merge with coordinates and impute
                df_coord_att = pd.merge(coords, df_attention, on="location", how="left")
                # imputed_coord_att = self.spatial_processor.impute_nearby(df_coord_att)

                # Save results
                output_file = f"{output_path}_{season}.csv"
                df_coord_att.to_csv(output_file, index=False)
                logger.info(f"Saved {season} results to {output_file}")


def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate driver attention maps using per-pixel phenology.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--PFT", type=str, help="Plant functional type")
    parser.add_argument(
        "--pred_path", type=str, required=True, help="Path to predictions pickle file"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data pickle file"
    )
    parser.add_argument(
        "--coord_path", type=str, required=True, help="Path to coordinates parquet file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path prefix for results"
    )
    parser.add_argument(
        "--forecast_window_length", type=int, required=True, help="Forecast window size"
    )
    parser.add_argument(
        "--test_mode", action="store_true", help="Enable test mode for debugging"
    )
    args = parser.parse_args()

    try:
        # Initialize and run workflow
        workflow = AnalysisWorkflow()
        workflow.run_analysis(
            data_path=args.data_path,
            pred_path=args.pred_path,
            coord_path=args.coord_path,
            output_path=args.output_path,
            forecast_window=args.forecast_window_length,
            pft=args.PFT,
            test_mode=args.test_mode,
        )

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
