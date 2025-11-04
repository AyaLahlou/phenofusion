"""
Refactored Driver Data Extraction Module

This module extracts climate driver importance from TFT model predictions
and generates spatial driver maps for phenological analysis.

Key improvements:
- Robust phenology detection using CSIF slope analysis
- Flexible latitude-based season detection
- Better handling of missing data and edge cases
- Improved spatial imputation
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from scipy.stats import linregress
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DriverDataExtractor:
    """Extract climate driver importance from TFT predictions."""

    def __init__(
        self,
        data_path: str,
        pred_path: str,
        coord_path: str,
        forecast_window: int = 30,
        pft: Optional[str] = None,
    ):
        """
        Initialize the driver data extractor.

        Args:
            data_path: Path to original processed data pickle
            pred_path: Path to predictions pickle
            coord_path: Path to coordinates parquet
            forecast_window: Length of forecast window in days
            pft: Plant functional type (e.g., 'BET', 'BDT')
        """
        self.data_path = data_path
        self.pred_path = pred_path
        self.coord_path = coord_path
        self.forecast_window = forecast_window
        self.pft = pft

        # Load data
        logger.info(f"Loading data from {data_path}")
        with open(data_path, "rb") as fp:
            self.data = pickle.load(fp)

        logger.info(f"Loading predictions from {pred_path}")
        with open(pred_path, "rb") as fp:
            self.preds = pickle.load(fp)

        logger.info(f"Loading coordinates from {coord_path}")
        self.coords = pd.read_parquet(coord_path).drop_duplicates()

        # Set phenology detection thresholds based on PFT
        self.min_diff = 0.05 if pft in ["BET", "SHR"] else 0.20
        self.min_slope = 0.001 if pft in ["BET", "SHR"] else 0.002

        logger.info(f"Initialized for PFT: {pft}")
        logger.info(
            f"Phenology thresholds - min_diff: {self.min_diff}, min_slope: {self.min_slope}"
        )

    def get_analysis_df(self) -> pd.DataFrame:
        """
        Create analysis DataFrame matching data to predictions and coordinates.

        Returns:
            DataFrame with predictions, observations, and coordinates
        """
        logger.info("Creating analysis DataFrame")

        # Extract test data
        test_data = self.data["data_sets"]["test"]

        # Create base DataFrame
        df = pd.DataFrame(
            {
                "Index": test_data["id"].flatten(),
                "CSIF": test_data["target"].flatten(),
            }
        )

        # Add predictions
        df["pred_05"] = self.preds["predicted_quantiles"][:, :, 1].flatten()

        # Parse location and time
        df[["location", "time"]] = df["Index"].str.split("_", n=1, expand=True)
        df["location"] = df["location"].astype(int)
        df["time"] = pd.to_datetime(df["time"])

        # Sort and add temporal features
        df = df.sort_values(by=["location", "time"])
        df["doy"] = df["time"].dt.dayofyear
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day

        # Drop index column
        df = df.drop(columns=["Index"])

        # Merge with coordinates
        df = pd.merge(self.coords, df, on="location", how="left")

        logger.info(f"Created analysis DataFrame with {len(df)} records")
        logger.info(f"Unique locations: {df['location'].nunique()}")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")

        return df

    def detect_phenology_indices(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Detect SOS and EOS samples using CSIF slope analysis.

        Args:
            df: Analysis DataFrame with CSIF time series

        Returns:
            Tuple of (SOS_indices, EOS_indices)
        """
        logger.info("Detecting phenology indices using slope analysis")

        SOS_indices = []
        EOS_indices = []

        # Iterate over DataFrame in batches
        batch_size = self.forecast_window

        sos_count = 0
        eos_count = 0

        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]

            # Skip incomplete batches
            if len(batch_df) < batch_size:
                continue

            # Check if batch is from same location
            if batch_df["location"].nunique() > 1:
                continue

            # Get CSIF values
            csif_values = batch_df["CSIF"].values

            # Check if there's sufficient signal
            csif_range = abs(csif_values[-1] - csif_values[0])
            if csif_range < self.min_diff:
                continue

            # Calculate slope
            x = np.arange(len(csif_values))
            slope, _, _, _, _ = linregress(x, csif_values)

            # Classify as SOS or EOS based on slope
            if slope >= self.min_slope:
                # Positive slope = Start of Season
                SOS_indices.append(batch_df.index[0])
                sos_count += 1
            elif slope <= -self.min_slope - 0.0005:
                # Negative slope = End of Season
                EOS_indices.append(batch_df.index[0])
                eos_count += 1

        # Convert to prediction indices
        SOS_pred_indices = [int(i / batch_size) for i in SOS_indices]
        EOS_pred_indices = [int(i / batch_size) for i in EOS_indices]

        # Filter out indices beyond prediction array bounds
        max_pred_index = len(self.preds["attention_scores"]) - 1
        SOS_pred_indices = [idx for idx in SOS_pred_indices if idx <= max_pred_index]
        EOS_pred_indices = [idx for idx in EOS_pred_indices if idx <= max_pred_index]

        logger.info(
            f"Detected {len(SOS_pred_indices)} SOS samples (from {sos_count} raw)"
        )
        logger.info(
            f"Detected {len(EOS_pred_indices)} EOS samples (from {eos_count} raw)"
        )

        return SOS_pred_indices, EOS_pred_indices

    def find_max_attention_window(self, index: int) -> int:
        """
        Find the time window with maximum attention scores.

        Args:
            index: Sample index in predictions

        Returns:
            Start index of maximum attention window
        """
        # Get mean attention across all horizons
        att_array = np.mean(self.preds["attention_scores"][index], axis=0)

        max_sum = -np.inf
        best_start_index = None

        # Slide window to find maximum
        max_start = len(att_array) - self.forecast_window
        for i in range(max_start):
            current_sum = np.sum(att_array[i : i + self.forecast_window])
            if current_sum > max_sum:
                max_sum = current_sum
                best_start_index = i

        return best_start_index if best_start_index is not None else 0

    def extract_driver_weights(self, indices: List[int]) -> pd.DataFrame:
        """
        Extract climate driver attention weights for given sample indices.

        Args:
            indices: List of prediction sample indices

        Returns:
            DataFrame with location and driver weights
        """
        logger.info(f"Extracting driver weights for {len(indices)} samples")

        driver_data = []
        max_window_start = 365 - self.forecast_window

        for index in indices:
            try:
                # Find maximum attention window
                window_start = self.find_max_attention_window(index)

                # Skip if window extends beyond valid range
                if window_start > max_window_start:
                    continue

                # Extract median weights for each driver
                weights = {}
                hist_weights = self.preds["historical_selection_weights"][index]

                for i, var in enumerate(
                    ["tmin", "tmax", "rad", "precip", "photo", "sm"], 1
                ):
                    weights[f"hist_{var}"] = np.median(
                        hist_weights[
                            window_start : window_start + self.forecast_window, i
                        ]
                    )

                # Get location ID
                location_id = int(
                    self.data["data_sets"]["test"]["id"][index][0].split("_")[0]
                )
                weights["location"] = location_id

                driver_data.append(weights)

            except Exception as e:
                logger.warning(f"Error processing index {index}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(driver_data)

        logger.info(f"Extracted weights for {len(df)} locations")

        return df

    def impute_nearby_values(
        self,
        df: pd.DataFrame,
        lat_range: float = 0.5,
        lon_range: float = 0.5,
        max_distance: float = 2.0,
    ) -> pd.DataFrame:
        """
        Impute missing driver values using nearby spatial locations.

        Args:
            df: DataFrame with coordinates and driver weights
            lat_range: Latitude search range in degrees
            lon_range: Longitude search range in degrees
            max_distance: Maximum distance for imputation in degrees

        Returns:
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

        # Identify rows with missing data
        missing_mask = df[driver_cols].isnull().all(axis=1)
        n_missing = missing_mask.sum()

        if n_missing == 0:
            logger.info("No missing values to impute")
            return df

        logger.info(f"Found {n_missing} locations with missing data")

        # Create a copy to avoid modifying during iteration
        df_result = df.copy()

        # For each row with data, find and fill nearby missing rows
        for idx, row in df.iterrows():
            if row[driver_cols].isnull().any():
                continue

            # Find nearby locations with missing data
            lat_match = (df["latitude"] >= row["latitude"] - lat_range) & (
                df["latitude"] <= row["latitude"] + lat_range
            )
            lon_match = (df["longitude"] >= row["longitude"] - lon_range) & (
                df["longitude"] <= row["longitude"] + lon_range
            )

            # Additional distance check
            if "latitude" in df.columns and "longitude" in df.columns:
                lat_diff = np.abs(df["latitude"] - row["latitude"])
                lon_diff = np.abs(df["longitude"] - row["longitude"])
                dist = np.sqrt(lat_diff**2 + lon_diff**2)
                dist_match = dist <= max_distance
            else:
                dist_match = True

            # Combined mask
            nearby_missing = lat_match & lon_match & dist_match & missing_mask

            if nearby_missing.any():
                # Impute values
                df_result.loc[nearby_missing, driver_cols] = row[driver_cols].values

        # Count remaining missing values
        n_remaining = df_result[driver_cols].isnull().all(axis=1).sum()
        logger.info(
            f"After imputation: {n_remaining} locations still missing ({n_missing - n_remaining} filled)"
        )

        return df_result

    def save_driver_data(self, indices: List[int], output_path: str):
        """
        Extract driver weights and save to CSV with spatial imputation.

        Args:
            indices: List of prediction sample indices
            output_path: Path to save CSV file
        """
        logger.info(f"Saving driver data to {output_path}")

        # Extract driver weights
        driver_df = self.extract_driver_weights(indices)

        if len(driver_df) == 0:
            logger.warning("No driver data extracted - creating empty file")
            driver_df = pd.DataFrame(
                columns=[
                    "location",
                    "hist_tmin",
                    "hist_tmax",
                    "hist_rad",
                    "hist_precip",
                    "hist_photo",
                    "hist_sm",
                ]
            )

        # Merge with coordinates
        coord_driver_df = pd.merge(self.coords, driver_df, on="location", how="left")

        # Apply spatial imputation
        imputed_df = self.impute_nearby_values(coord_driver_df)

        # Save to CSV
        imputed_df.to_csv(output_path, index=False)

        logger.info(f"Saved driver data with {len(imputed_df)} locations")

        # Log data quality metrics
        driver_cols = [
            "hist_tmin",
            "hist_tmax",
            "hist_rad",
            "hist_precip",
            "hist_photo",
            "hist_sm",
        ]
        n_complete = (~imputed_df[driver_cols].isnull().any(axis=1)).sum()
        logger.info(
            f"Complete records: {n_complete}/{len(imputed_df)} ({100*n_complete/len(imputed_df):.1f}%)"
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract climate driver importance from TFT predictions"
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
        "--output_path", type=str, required=True, help="Base path for output CSV files"
    )
    parser.add_argument(
        "--forecast_window_length",
        type=int,
        default=30,
        help="Forecast window size in days",
    )

    args = parser.parse_args()

    try:
        # Initialize extractor
        extractor = DriverDataExtractor(
            data_path=args.data_path,
            pred_path=args.pred_path,
            coord_path=args.coord_path,
            forecast_window=args.forecast_window_length,
            pft=args.PFT,
        )

        # Get analysis DataFrame
        df = extractor.get_analysis_df()

        # Detect phenology indices
        SOS_indices, EOS_indices = extractor.detect_phenology_indices(df)

        # Save driver data for both seasons
        logger.info("Processing SOS (Start of Season) data")
        extractor.save_driver_data(SOS_indices, f"{args.output_path}_SOS.csv")

        logger.info("Processing EOS (End of Season) data")
        extractor.save_driver_data(EOS_indices, f"{args.output_path}_EOS.csv")

        logger.info("Driver data extraction complete!")

    except Exception as e:
        logger.error(f"Error during driver data extraction: {e}")
        raise


if __name__ == "__main__":
    main()
