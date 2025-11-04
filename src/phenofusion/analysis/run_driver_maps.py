#!/usr/bin/env python3
"""
Refactored Driver Map Generator

This script generates RGB driver maps from extracted climate driver data,
with improved handling of missing data and better spatial interpolation.

Key improvements:
- Robust handling of sparse data
- Better spatial interpolation for missing values
- Quality metrics and diagnostics
- Support for partial coverage maps
"""

import sys
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.interpolate import griddata
from itertools import product

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImprovedDriverMapGenerator:
    """Generate driver maps with improved handling of sparse data."""

    def __init__(self, output_dir: str):
        """
        Initialize the driver map generator.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized driver map generator")
        logger.info(f"Output directory: {self.output_dir}")

    def load_and_validate_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV file and validate data quality.

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded and validated DataFrame
        """
        logger.info(f"Loading {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Check required columns
            required_cols = [
                "latitude",
                "longitude",
                "hist_tmin",
                "hist_tmax",
                "hist_rad",
                "hist_precip",
                "hist_photo",
                "hist_sm",
            ]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Log data quality
            total_rows = len(df)
            driver_cols = [
                "hist_tmin",
                "hist_tmax",
                "hist_rad",
                "hist_precip",
                "hist_photo",
                "hist_sm",
            ]

            complete_rows = (~df[driver_cols].isnull().any(axis=1)).sum()
            logger.info(
                f"Loaded {total_rows} rows, {complete_rows} complete ({100*complete_rows/total_rows:.1f}%)"
            )

            # Check coordinate coverage
            lat_range = (df["latitude"].min(), df["latitude"].max())
            lon_range = (df["longitude"].min(), df["longitude"].max())
            logger.info(f"Latitude range: {lat_range[0]:.2f} to {lat_range[1]:.2f}")
            logger.info(f"Longitude range: {lon_range[0]:.2f} to {lon_range[1]:.2f}")

            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def interpolate_missing_values(
        self, df: pd.DataFrame, method: str = "nearest"
    ) -> pd.DataFrame:
        """
        Interpolate missing driver values using spatial interpolation.

        Args:
            df: DataFrame with coordinates and driver values
            method: Interpolation method ('nearest', 'linear', 'cubic')

        Returns:
            DataFrame with interpolated values
        """
        logger.info(f"Interpolating missing values using {method} method")

        driver_cols = [
            "hist_tmin",
            "hist_tmax",
            "hist_rad",
            "hist_precip",
            "hist_photo",
            "hist_sm",
        ]

        df_result = df.copy()

        for col in driver_cols:
            # Get rows with valid data for this column
            valid_mask = ~df[col].isnull()

            if valid_mask.sum() < 3:
                logger.warning(
                    f"Insufficient data for interpolation of {col} ({valid_mask.sum()} points)"
                )
                continue

            # Extract valid points
            valid_points = df.loc[valid_mask, ["latitude", "longitude"]].values
            valid_values = df.loc[valid_mask, col].values

            # Find missing points
            missing_mask = df[col].isnull()
            if not missing_mask.any():
                continue

            missing_points = df.loc[missing_mask, ["latitude", "longitude"]].values

            try:
                # Interpolate
                interpolated_values = griddata(
                    valid_points, valid_values, missing_points, method=method
                )

                # Fill interpolated values
                df_result.loc[missing_mask, col] = interpolated_values

                n_filled = (~np.isnan(interpolated_values)).sum()
                logger.info(
                    f"  {col}: Filled {n_filled}/{len(missing_points)} missing values"
                )

            except Exception as e:
                logger.warning(f"  Error interpolating {col}: {e}")
                continue

        return df_result

    def create_global_grid(
        self, df: pd.DataFrame, interpolate: bool = True
    ) -> pd.DataFrame:
        """
        Create a global grid with normalized driver values.

        Args:
            df: DataFrame with driver data
            interpolate: Whether to interpolate missing values

        Returns:
            DataFrame on global grid with normalized drivers
        """
        logger.info("Creating global grid")

        # Apply interpolation if requested
        if interpolate:
            df = self.interpolate_missing_values(df)

        # Group by coordinates and average
        grouped_df = df.groupby(["latitude", "longitude"]).mean().reset_index()

        # Create composite driver variables
        grouped_df["hist_temp"] = grouped_df["hist_tmin"] + grouped_df["hist_tmax"]
        grouped_df["hist_sol"] = grouped_df["hist_rad"] + grouped_df["hist_photo"]
        grouped_df["hist_p"] = grouped_df["hist_precip"] + grouped_df["hist_sm"]

        # Normalize drivers
        driver_cols = ["hist_temp", "hist_sol", "hist_p"]
        total = grouped_df[driver_cols].sum(axis=1)

        # Add small epsilon to avoid division by zero
        total = total + 1e-10

        for col in driver_cols:
            grouped_df[col] = grouped_df[col] / total

        # Create full global coordinate grid
        lat_values = np.linspace(90.0, -89.75, 720)
        lon_values = np.linspace(-180.0, 179.75, 1440)
        combinations = list(product(lat_values, lon_values))
        df_fullcoord = pd.DataFrame(combinations, columns=["latitude", "longitude"])

        # Merge with full grid
        full_df = pd.merge(
            df_fullcoord, grouped_df, on=["latitude", "longitude"], how="left"
        )

        # Log coverage
        coverage = (~full_df[driver_cols].isnull().any(axis=1)).sum()
        total_points = len(full_df)
        logger.info(
            f"Grid coverage: {coverage}/{total_points} points ({100*coverage/total_points:.1f}%)"
        )

        return full_df

    def colorize_drivers(self, df: pd.DataFrame) -> List[np.ndarray]:
        """
        Convert normalized driver data to RGB color arrays.

        Args:
            df: DataFrame with normalized driver data

        Returns:
            List of RGB arrays [B, G, R] for [precip, solar, temp]
        """
        logger.info("Converting drivers to RGB colors")

        # Reshape to 2D grids
        temp_data = np.reshape(df["hist_temp"].values, (720, 1440))
        sol_data = np.reshape(df["hist_sol"].values, (720, 1440))
        precip_data = np.reshape(df["hist_p"].values, (720, 1440))

        # Scale to 0-255 range
        w = 255
        scale = 100

        def process_channel(data):
            """Process a single channel."""
            color_data = data * w / scale
            channel = []
            for row in color_data:
                processed_row = []
                for val in row:
                    if np.isnan(val):
                        processed_row.append(255)  # White for missing data
                    else:
                        processed_row.append(np.clip(val, 0, 255))
                channel.append(processed_row)
            return channel

        # Process each channel
        # Order: [precip, solar, temp] -> [B, G, R] -> [cyan, magenta, yellow]
        b_channel = process_channel(precip_data)
        g_channel = process_channel(sol_data)
        r_channel = process_channel(temp_data)

        return [b_channel, g_channel, r_channel]

    def plot_driver_map(
        self,
        rgb_list: List[np.ndarray],
        title: str,
        projection: str = "PlateCarree",
        crop_lat: float = -60,
        dpi: int = 300,
        figsize: Tuple[int, int] = (16, 8),
    ):
        """
        Create and save a driver map.

        Args:
            rgb_list: List of RGB channel arrays
            title: Plot title and filename
            projection: Map projection ('PlateCarree' or 'Robinson')
            crop_lat: Latitude below which to crop
            dpi: Resolution for saved image
            figsize: Figure size tuple
        """
        logger.info(f"Creating map: {title}")

        # Validate input
        if len(rgb_list) != 3:
            raise ValueError("rgb_list must contain exactly 3 channels (R, G, B)")

        # Stack RGB channels
        rgb_data = np.stack(rgb_list, axis=-1)

        if rgb_data.ndim != 3 or rgb_data.shape[-1] != 3:
            raise ValueError("Stacked RGB data must have shape (H, W, 3)")

        # Choose projection
        if projection.lower() == "robinson":
            proj = ccrs.Robinson()
        else:
            proj = ccrs.PlateCarree()

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        ax.set_global()

        # Add cyclic point for seamless plotting
        lon = np.linspace(-180.0, 180.0, rgb_data.shape[1], endpoint=False)
        rgb_cyc, lon_cyc = add_cyclic_point(rgb_data, coord=lon, axis=1)

        # Create alpha mask to hide regions below crop_lat
        H, W_cyc, _ = rgb_cyc.shape
        lats = np.linspace(90.0, -90.0, H)[:, None]
        alpha = np.repeat((lats >= crop_lat).astype(float), W_cyc, axis=1)

        # Plot RGB data
        ax.imshow(
            rgb_cyc,
            extent=[lon_cyc.min(), lon_cyc.max(), -90.0, 90.0],
            origin="upper",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
            alpha=alpha,
        )

        # Crop view if specified
        if crop_lat is not None:
            ax.set_extent(
                [lon_cyc.min(), lon_cyc.max(), crop_lat, 90.0], crs=ccrs.PlateCarree()
            )

        # Add map features
        ax.coastlines(resolution="110m", color="black", linewidth=0.5)
        ax.gridlines(draw_labels=False, linestyle="--", color="gray", alpha=0.3)

        # Add title
        if title:
            plt.title(title, fontsize=14, fontweight="bold", pad=20)

        # Add legend
        legend_text = "Drivers: Red=Temperature, Green=Solar, Blue=Precipitation"
        plt.text(
            0.5,
            -0.05,
            legend_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            style="italic",
        )

        # Save plot
        output_path = self.output_dir / f"{title.replace(' ', '_')}.png"
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        logger.info(f"Saved map to {output_path}")

        plt.close(fig)

    def generate_map_from_csv(
        self, csv_path: str, map_name: str, interpolate: bool = True, **plot_kwargs
    ):
        """
        Generate a driver map from a CSV file.

        Args:
            csv_path: Path to CSV file with driver data
            map_name: Name for the output map
            interpolate: Whether to interpolate missing values
            **plot_kwargs: Additional plotting arguments
        """
        # Load data
        df = self.load_and_validate_csv(csv_path)

        if df.empty:
            logger.error(f"No valid data loaded from {csv_path}")
            return

        # Create global grid
        full_df = self.create_global_grid(df, interpolate=interpolate)

        # Colorize
        rgb_list = self.colorize_drivers(full_df)

        # Plot
        self.plot_driver_map(rgb_list, map_name, **plot_kwargs)


def discover_csv_files(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Discover available CSV files in a directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Dictionary mapping PFT and season to file paths
    """
    data_path = Path(data_dir)
    file_mapping = {}

    # Find all CSV files
    csv_files = list(data_path.glob("*_SOS.csv")) + list(data_path.glob("*_EOS.csv"))

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    for csv_file in csv_files:
        filename = csv_file.stem  # Remove .csv extension

        # Determine season
        if filename.endswith("_SOS"):
            season = "SOS"
            pft_name = filename[:-4]  # Remove _SOS
        elif filename.endswith("_EOS"):
            season = "EOS"
            pft_name = filename[:-4]  # Remove _EOS
        else:
            continue

        # Store mapping
        if pft_name not in file_mapping:
            file_mapping[pft_name] = {}

        file_mapping[pft_name][season] = str(csv_file)
        logger.info(f"  Found: {pft_name} - {season}")

    return file_mapping


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate driver maps from CSV files")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing CSV files with driver data",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save output maps"
    )
    parser.add_argument(
        "--interpolate", action="store_true", help="Interpolate missing values"
    )
    parser.add_argument(
        "--projection",
        choices=["PlateCarree", "Robinson"],
        default="PlateCarree",
        help="Map projection",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved images"
    )
    parser.add_argument(
        "--crop-lat", type=float, default=-60, help="Latitude below which to crop"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = ImprovedDriverMapGenerator(args.output_dir)

    # Discover CSV files
    file_mapping = discover_csv_files(args.data_dir)

    if not file_mapping:
        logger.error("No CSV files found!")
        sys.exit(1)

    # Plot settings
    plot_kwargs = {
        "projection": args.projection,
        "dpi": args.dpi,
        "crop_lat": args.crop_lat,
    }

    success_count = 0
    total_count = 0

    # Generate maps for each PFT and season
    for pft, seasons in file_mapping.items():
        for season, csv_path in seasons.items():
            total_count += 1
            map_name = f"{pft}_{season}"

            try:
                logger.info(f"\n=== Generating map: {map_name} ===")
                generator.generate_map_from_csv(
                    csv_path, map_name, interpolate=args.interpolate, **plot_kwargs
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to generate {map_name}: {e}")
                continue

    # Summary
    logger.info("\n=== Generation Complete ===")
    logger.info(f"Successfully generated {success_count}/{total_count} maps")
    logger.info(f"Output saved to: {args.output_dir}")

    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
