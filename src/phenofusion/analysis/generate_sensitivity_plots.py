#!/usr/bin/env python3
"""
Sensitivity Analysis and Temporal Comparison for Phenological Drivers

This script analyzes temporal changes in climate driver sensitivity by comparing
attention weights between different years. It generates difference maps showing
how the relative importance of temperature, solar radiation, and precipitation
has changed over time.

The script processes model predictions and attention weights to create visualizations
that highlight temporal shifts in environmental driver sensitivity across different
plant functional types and phenological phases.

Usage:
    python generate_sensitivity_plots.py --config config.yaml

Author: Refactored from sensitivity_plot.py
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
from dataclasses import dataclass
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime
from itertools import product

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for sensitivity analysis."""

    data_directory: str
    pred_directory: str
    coord_directory: str
    output_directory: str
    years: List[int]
    seasons: List[str]
    cluster_names: List[str]
    lat_range: float = 0.5
    lon_range: float = 0.5


class PhenologicalDataProcessor:
    """Process phenological data and model predictions."""

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the data processor.

        Args:
            config: Analysis configuration object
        """
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Seasonal timing dictionaries for different latitude bands
        self.eos_timing = {
            (60, 90): (7, 8),
            (35, 60): (8, 9),
            (30, 35): (8, 9),
            (20, 30): (9, 9),
            (15, 20): (9, 10),
            (5, 15): (9, 10),
            (0, 5): (7, 9),
            (-3, 0): (5, 6),
            (-5, -3): (4, 5),
            (-10, -5): (4, 5),
            (-24, -10): (3, 4),
            (-30, -24): (4, 5),
            (-40, -30): (7, 8),
            (-50, -40): (4, 5),
            (-61, -50): (5, 6),
        }

        self.sos_timing = {
            (60, 90): (5, 6),
            (55, 60): (4, 5),
            (40, 55): (3, 5),
            (30, 40): (4, 6),
            (20, 30): (5, 6),
            (15, 20): (6, 7),
            (12, 15): (5, 6),
            (10, 12): (4, 5),
            (7, 10): (3, 4),
            (3, 7): (2, 4),
            (0, 3): (3, 4),
            (-2, 0): (5, 6),
            (-5, -2): (6, 7),
            (-13, -5): (8, 9),
            (-20, -13): (9, 11),
            (-26, -20): (9, 10),
            (-30, -26): (8, 9),
            (-40, -30): (7, 9),
            (-50, -40): (7, 8),
            (-61, -50): (8, 9),
        }

    def load_prediction_data(self, data_path: str, pred_path: str) -> Tuple[Dict, Dict]:
        """
        Load data and prediction files.

        Args:
            data_path: Path to data pickle file
            pred_path: Path to prediction pickle file

        Returns:
            Tuple of (data, predictions) dictionaries
        """
        try:
            with open(data_path, "rb") as fp:
                data = pickle.load(fp)

            with open(pred_path, "rb") as fp:
                preds = pickle.load(fp)

            logger.info(f"Loaded data from {data_path} and {pred_path}")
            return data, preds

        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            return {}, {}

    def get_analysis_dataframe(
        self, data: Dict, preds: Dict, coord_path: str
    ) -> pd.DataFrame:
        """
        Create analysis dataframe matching data to predictions and coordinates.

        Args:
            data: Data dictionary from pickle file
            preds: Predictions dictionary from pickle file
            coord_path: Path to coordinates parquet file

        Returns:
            Combined DataFrame with data, predictions, and coordinates
        """
        try:
            # Load coordinates
            coords = pd.read_parquet(coord_path).drop_duplicates()

            # Extract data and predictions
            df = pd.DataFrame(
                {
                    "Index": data["data_sets"]["test"]["id"].flatten(),
                    "CSIF": data["data_sets"]["test"]["target"].flatten(),
                    "pred_05": preds["predicted_quantiles"][:, :, 1].flatten(),
                }
            )

            # Parse location and time information
            df[["location_id", "time_id"]] = df["Index"].str.split(
                "_", n=1, expand=True
            )
            df["location_id"] = df["location_id"].astype(int)
            df["time_id"] = pd.to_datetime(df["time_id"])
            df = df.sort_values(by=["location_id", "time_id"])

            # Add temporal features
            df["doy"] = df["time_id"].dt.dayofyear
            df["year"] = df["time_id"].dt.year
            df["month"] = df["time_id"].dt.month
            df["day"] = df["time_id"].dt.day

            # Rename and clean columns
            df = df.rename(columns={"location_id": "location", "time_id": "time"})
            df = df.drop(columns=["Index"])

            # Merge with coordinates
            df = pd.merge(coords, df, on="location", how="left")

            logger.info(f"Created analysis DataFrame with {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error creating analysis DataFrame: {e}")
            return pd.DataFrame()

    def get_seasonal_indices(
        self,
        df: pd.DataFrame,
        month_start: int,
        month_end: int,
        specific_year: Optional[int] = None,
    ) -> List[int]:
        """
        Get indices for time series within a specific seasonal window.

        Args:
            df: Analysis DataFrame
            month_start: Start month for seasonal window
            month_end: End month for seasonal window
            specific_year: Specific year to analyze (if None, uses 2002-2010)

        Returns:
            List of indices for seasonal time series
        """
        res_indices = []
        year_range = range(2002, 2011) if specific_year is None else [specific_year]

        for year in year_range:
            dt_start = datetime(year, month_start, 10)
            dt_end = datetime(year, month_end, 30)
            df_current_year = df[df["year"] == year]

            # Filter by date range
            filtered_df = df_current_year.groupby("location").filter(
                lambda x: (x["time"].iloc[0] >= dt_start)
                and (x["time"].iloc[0] <= dt_end)
            )

            indices_array = np.unique(filtered_df[::30].index.to_numpy())
            res_indices.extend(np.round(indices_array / 30).astype(int))

        logger.info(
            f"Found {len(res_indices)} seasonal indices for {month_start}-{month_end}"
        )
        return res_indices

    def find_max_attention_window(self, preds: Dict, index: int) -> int:
        """
        Find the time window with maximum attention scores.

        Args:
            preds: Predictions dictionary
            index: Sample index

        Returns:
            Start index of maximum attention window
        """
        # Get mean attention across all horizons
        att_array = np.mean(preds["attention_scores"][index], axis=0)

        max_sum = -np.inf
        best_start_index = None

        # Slide 30-day window to find maximum attention
        for i in range(396 - 30):
            current_sum = np.sum(att_array[i : i + 30])
            if current_sum > max_sum:
                max_sum = current_sum
                best_start_index = i

        return best_start_index

    def extract_attention_weights(
        self, data: Dict, preds: Dict, coord_path: str, year: int, season: str
    ) -> pd.DataFrame:
        """
        Extract attention weights for climate drivers.

        Args:
            data: Data dictionary
            preds: Predictions dictionary
            coord_path: Path to coordinates file
            year: Year to analyze
            season: Season ('sos' or 'eos')

        Returns:
            DataFrame with attention weights by location
        """
        df = self.get_analysis_dataframe(data, preds, coord_path)

        # Select timing dictionary based on season
        timing_dict = self.sos_timing if season == "sos" else self.eos_timing

        select_indices = []

        # Get indices for each latitude band
        for lats, months in timing_dict.items():
            df_current = df[(df["latitude"] > lats[0]) & (df["latitude"] <= lats[1])]
            indices = self.get_seasonal_indices(df_current, months[0], months[1], year)
            select_indices.extend(indices)

        # Initialize attention weights DataFrame
        attention_df = pd.DataFrame(
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

        for index in select_indices:
            window_start = self.find_max_attention_window(preds, index)

            if window_start <= 335:  # Valid window
                # Extract median weights for each variable
                weights = {}
                for i, var in enumerate(
                    [
                        "hist_tmin",
                        "hist_tmax",
                        "hist_rad",
                        "hist_precip",
                        "hist_photo",
                        "hist_sm",
                    ],
                    1,
                ):
                    weights[var] = np.median(
                        preds["historical_selection_weights"][
                            index, window_start : window_start + 30, i
                        ]
                    )

                location = int(data["data_sets"]["test"]["id"][index][0].split("_")[0])
                weights["location"] = location

                new_row = pd.DataFrame([weights])
                attention_df = pd.concat([attention_df, new_row], ignore_index=True)

        # Merge with coordinates and apply spatial imputation
        coords = pd.read_parquet(coord_path).drop_duplicates()
        coord_att_df = pd.merge(coords, attention_df, on="location", how="left")
        imputed_df = self.impute_nearby_values(coord_att_df)

        logger.info(f"Extracted attention weights for {len(imputed_df)} locations")
        return imputed_df

    def impute_nearby_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using nearby spatial locations.

        Args:
            df: DataFrame with spatial coordinates and attention weights

        Returns:
            DataFrame with imputed values
        """
        driver_cols = [
            "hist_tmin",
            "hist_tmax",
            "hist_rad",
            "hist_precip",
            "hist_photo",
            "hist_sm",
        ]

        for i, row in df.iterrows():
            if not row[driver_cols].isnull().all():
                # Find nearby locations with missing values
                mask = (
                    (df["latitude"] >= row["latitude"] - self.config.lat_range)
                    & (df["latitude"] <= row["latitude"] + self.config.lat_range)
                    & (df["longitude"] >= row["longitude"] - self.config.lon_range)
                    & (df["longitude"] <= row["longitude"] + self.config.lon_range)
                    & df[driver_cols].isnull().all(axis=1)
                )

                # Impute missing values
                df.loc[mask, driver_cols] = row[driver_cols].values

        return df


class SensitivityAnalyzer:
    """Analyze and visualize temporal sensitivity changes."""

    def __init__(self, processor: PhenologicalDataProcessor):
        """
        Initialize the sensitivity analyzer.

        Args:
            processor: Data processor instance
        """
        self.processor = processor
        self.config = processor.config

    def concatenate_dataframes(self, df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate and process multiple DataFrames for mapping.

        Args:
            df_list: List of DataFrames to concatenate

        Returns:
            Processed DataFrame with normalized driver values
        """
        if not df_list:
            logger.warning("No dataframes to concatenate")
            return pd.DataFrame()

        # Combine and group by coordinates
        df = pd.concat(df_list, ignore_index=True)
        grouped_df = df.groupby(["latitude", "longitude"]).mean().reset_index()

        # Create composite driver variables
        grouped_df["hist_temp"] = grouped_df["hist_tmin"] + grouped_df["hist_tmax"]
        grouped_df["hist_sol"] = grouped_df["hist_rad"] + grouped_df["hist_photo"]
        grouped_df["hist_p"] = grouped_df["hist_precip"] + grouped_df["hist_sm"]

        # Normalize driver variables
        driver_cols = ["hist_temp", "hist_sol", "hist_p"]
        df_normalized = grouped_df[driver_cols].div(
            grouped_df[driver_cols].sum(axis=1), axis=0
        )
        grouped_df[driver_cols] = df_normalized

        # Create full coordinate grid
        lat_values = np.linspace(90.0, -89.75, 720)
        lon_values = np.linspace(-180.0, 179.75, 1440)
        combinations = list(product(lat_values, lon_values))
        df_fullcoord = pd.DataFrame(combinations, columns=["latitude", "longitude"])

        # Merge with full grid
        full_df = pd.merge(
            df_fullcoord, grouped_df, on=["latitude", "longitude"], how="left"
        )

        logger.info(f"Created full coordinate grid with {len(full_df)} points")
        return full_df

    def process_year_data(self, year: int, season: str) -> pd.DataFrame:
        """
        Process data for a specific year and season.

        Args:
            year: Year to process
            season: Season ('sos' or 'eos')

        Returns:
            Processed DataFrame for the year
        """
        df_list = []

        for cluster in self.config.cluster_names:
            data_path = os.path.join(self.config.data_directory, f"{cluster}.pickle")
            pred_path = os.path.join(self.config.pred_directory, f"pred_{cluster}.pkl")
            coord_path = os.path.join(self.config.coord_directory, f"{cluster}.parquet")

            # Check if files exist
            if not all(os.path.exists(p) for p in [data_path, pred_path, coord_path]):
                logger.warning(f"Missing files for cluster {cluster}")
                continue

            try:
                # Load data and extract attention weights
                data, preds = self.processor.load_prediction_data(data_path, pred_path)
                if data and preds:
                    df_coord_att = self.processor.extract_attention_weights(
                        data, preds, coord_path, year, season
                    )
                    df_list.append(df_coord_att)

            except Exception as e:
                logger.error(f"Error processing cluster {cluster}: {e}")
                continue

        if df_list:
            return self.concatenate_dataframes(df_list)
        else:
            logger.error(f"No valid data found for year {year}, season {season}")
            return pd.DataFrame()

    def calculate_temporal_differences(
        self, year1: int, year2: int, season: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate temporal differences in driver sensitivity.

        Args:
            year1: First year (baseline)
            year2: Second year (comparison)
            season: Season to analyze

        Returns:
            Tuple of (temp_diff, solar_diff, precip_diff) arrays
        """
        logger.info(f"Calculating differences between {year1} and {year2} for {season}")

        # Process data for both years
        data_year1 = self.process_year_data(year1, season)
        data_year2 = self.process_year_data(year2, season)

        if data_year1.empty or data_year2.empty:
            logger.error("Failed to process data for one or both years")
            return np.array([]), np.array([]), np.array([])

        # Calculate differences
        difference = data_year1 - data_year2

        # Reshape to 2D grids
        temp_diff = np.reshape(difference["hist_temp"], (720, 1440))
        solar_diff = np.reshape(difference["hist_sol"], (720, 1440))
        precip_diff = np.reshape(difference["hist_p"], (720, 1440))

        logger.info("Successfully calculated temporal differences")
        return temp_diff, solar_diff, precip_diff

    def plot_difference_map(
        self,
        channel_data: np.ndarray,
        title: str,
        season: str,
        year1: int,
        year2: int,
        save: bool = True,
        show: bool = False,
        dpi: int = 300,
    ) -> None:
        """
        Plot a difference map for temporal sensitivity changes.

        Args:
            channel_data: 2D array of difference values
            title: Plot title and variable name
            season: Season identifier
            year1: First year
            year2: Second year
            save: Whether to save the plot
            show: Whether to display the plot
            dpi: Resolution for saved image
        """
        # Validate input
        channel_data = np.asarray(channel_data, dtype=np.float32)
        if channel_data.ndim != 2:
            raise ValueError("Input data must be a 2D array")

        # Create figure
        plt.figure(figsize=(16, 8))
        ax = plt.axes(projection=ccrs.Robinson())

        # Add map features
        ax.coastlines(resolution="110m", color="black", linewidth=0.5)
        ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.3)

        # Plot difference data with diverging colormap
        norm = TwoSlopeNorm(vcenter=0)
        img = plt.imshow(
            channel_data,
            extent=[-180, 180, -90, 90],
            origin="upper",
            norm=norm,
            cmap="RdBu_r",
            transform=ccrs.PlateCarree(),
            interpolation="none",
        )

        # Add colorbar
        cbar = plt.colorbar(
            img, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04, shrink=0.8
        )
        cbar.set_label(f"{title.title()} Sensitivity Difference", fontsize=12)

        # Add title
        plt.title(
            f"{title.title()} Sensitivity Change ({year1} - {year2})\n"
            f"During {season.upper()} Season",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Save plot
        if save:
            filename = f"{title}_sensitivity_difference_{year1}_{year2}_{season}.png"
            output_path = self.processor.output_dir / filename
            plt.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Saved plot to {output_path}")

        # Show or close
        if show:
            plt.show()
        else:
            plt.close()


def create_default_config() -> AnalysisConfig:
    """Create a default configuration for sensitivity analysis."""
    return AnalysisConfig(
        data_directory="/burg/glab/users/al4385/data/TFT_30_40years/",
        pred_directory="/burg/glab/users/al4385/predictions/TFT_30_40years/",
        coord_directory="/burg/glab/users/al4385/data/coordinates/",
        output_directory="./sensitivity_analysis_output/",
        years=[1985, 2020],
        seasons=["sos", "eos"],
        cluster_names=[
            "BDT_50_20",
            "BDT_-20_-60",
            "BDT_-20_20",
            "BDT_50_90",
            "BET",
            "NET",
            "NDT",
        ],
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate temporal sensitivity analysis plots"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/burg/glab/users/al4385/data/TFT_30_40years/",
        help="Directory containing data pickle files",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="/burg/glab/users/al4385/predictions/TFT_30_40years/",
        help="Directory containing prediction pickle files",
    )
    parser.add_argument(
        "--coord-dir",
        type=str,
        default="/burg/glab/users/al4385/data/coordinates/",
        help="Directory containing coordinate parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sensitivity_analysis_output/",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--year1", type=int, default=1985, help="First year for comparison (baseline)"
    )
    parser.add_argument(
        "--year2", type=int, default=2020, help="Second year for comparison"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["sos", "eos"],
        choices=["sos", "eos"],
        help="Seasons to analyze",
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots instead of just saving"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved images"
    )

    args = parser.parse_args()

    # Create configuration
    config = AnalysisConfig(
        data_directory=args.data_dir,
        pred_directory=args.pred_dir,
        coord_directory=args.coord_dir,
        output_directory=args.output_dir,
        years=[args.year1, args.year2],
        seasons=args.seasons,
        cluster_names=[
            "BDT_50_20",
            "BDT_-20_-60",
            "BDT_-20_20",
            "BDT_50_90",
            "BET",
            "NET",
            "NDT",
        ],
    )

    # Initialize processor and analyzer
    processor = PhenologicalDataProcessor(config)
    analyzer = SensitivityAnalyzer(processor)

    # Process each season
    for season in config.seasons:
        logger.info(f"\n=== Processing {season.upper()} season ===")

        # Calculate temporal differences
        temp_diff, solar_diff, precip_diff = analyzer.calculate_temporal_differences(
            args.year1, args.year2, season
        )

        if temp_diff.size == 0:
            logger.error(f"Failed to calculate differences for {season}")
            continue

        # Generate plots for each driver
        drivers = [
            (temp_diff, "temperature"),
            (solar_diff, "solar_radiation"),
            (precip_diff, "water_availability"),
        ]

        for diff_data, driver_name in drivers:
            analyzer.plot_difference_map(
                diff_data,
                driver_name,
                season,
                args.year1,
                args.year2,
                save=True,
                show=args.show_plots,
                dpi=args.dpi,
            )

    logger.info("\n=== Sensitivity analysis completed successfully! ===")


if __name__ == "__main__":
    main()
