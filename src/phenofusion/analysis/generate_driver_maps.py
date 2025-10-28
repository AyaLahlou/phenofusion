#!/usr/bin/env python3
"""
Generate Driver Maps for Phenological Analysis

This script creates and saves driver maps (RGB plots) showing the relative importance
of temperature, solar radiation, and precipitation drivers across different plant
functional types (PFTs) and phenological phases (SOS/EOS).

The script processes CSV files containing attention weights for different climate
variables and generates visualization maps using cartographic projections.

Usage:
    python generate_driver_maps.py --config config.yaml

Author: Refactored from drivermap_july2025.ipynb
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from itertools import product

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DriverMapGenerator:
    """Generate driver maps for phenological analysis."""

    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the driver map generator.

        Args:
            data_dir: Directory containing CSV files with driver data
            output_dir: Directory to save generated plots
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Latitude filtering for different PFTs
        self.lat_filters = {
            "BET": 30,  # Broadleaf Evergreen Trees
            "NET": 75,  # Needleleaf Evergreen Trees
            "NDT": 75,  # Needleleaf Deciduous Trees
            "SHR": 75,  # Shrubs
            "GRA": 75,  # Grasses
            "BDT_5020": (20, 50),  # Broadleaf Deciduous Trees (20-50N)
            "BDT_2020": (-20, 20),  # Broadleaf Deciduous Trees (-20-20)
            "BDT_2060": (-60, -20),  # Broadleaf Deciduous Trees (-60--20)
            "BDT_5090": (50, 90),  # Broadleaf Deciduous Trees (50-90N)
        }

    def load_csv_data(self, file_path: str, pft: str) -> pd.DataFrame:
        """
        Load and filter CSV data for a specific PFT.

        Args:
            file_path: Path to CSV file
            pft: Plant functional type identifier

        Returns:
            Filtered DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")

            # Apply latitude filtering based on PFT
            if pft in self.lat_filters:
                lat_filter = self.lat_filters[pft]
                if isinstance(lat_filter, tuple):
                    min_lat, max_lat = lat_filter
                    df = df[(df["latitude"] >= min_lat) & (df["latitude"] < max_lat)]
                else:
                    df = df[df["latitude"] < lat_filter]

                logger.info(f"After latitude filtering: {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

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

        # Combine dataframes and group by coordinates
        df = pd.concat(df_list, ignore_index=True)
        grouped_df = df.groupby(["latitude", "longitude"]).mean().reset_index()

        # Create composite driver variables
        grouped_df["hist_temp"] = grouped_df["hist_tmin"] + grouped_df["hist_tmax"]
        grouped_df["hist_sol"] = grouped_df["hist_rad"] + grouped_df["hist_photo"]
        grouped_df["hist_p"] = grouped_df["hist_precip"] + grouped_df["hist_sm"]

        # Normalize the driver variables
        columns_to_normalize = ["hist_temp", "hist_sol", "hist_p"]
        df_normalized = grouped_df[columns_to_normalize].div(
            grouped_df[columns_to_normalize].sum(axis=1), axis=0
        )

        grouped_df[columns_to_normalize] = df_normalized

        # Create full coordinate grid
        desired_lat_values = np.linspace(90.0, -89.75, 720)
        desired_lon_values = np.linspace(-180.0, 179.75, 1440)
        combinations = list(product(desired_lat_values, desired_lon_values))
        df_fullcoord = pd.DataFrame(combinations, columns=["latitude", "longitude"])

        # Merge with full coordinate grid
        full = pd.merge(
            df_fullcoord, grouped_df, on=["latitude", "longitude"], how="left"
        )

        logger.info(f"Created full grid with {len(full)} coordinate points")
        return full

    def colorize_data(self, df: pd.DataFrame) -> List[np.ndarray]:
        """
        Convert driver data to RGB color arrays.

        Args:
            df: DataFrame with normalized driver data

        Returns:
            List of RGB arrays [B, G, R] representing [precip, solar, temp]
        """
        # Reshape data to 2D grids
        temp_data = np.reshape(df["hist_temp"], (720, 1440))
        sol_data = np.reshape(df["hist_sol"], (720, 1440))
        precip_data = np.reshape(df["hist_p"], (720, 1440))

        # Convert to color values
        w = 255
        scale = 100

        # Process each channel
        channels = []
        for data, name in [
            (precip_data, "precip"),
            (sol_data, "solar"),
            (temp_data, "temp"),
        ]:
            color_data = data * w / scale

            channel = []
            for i in color_data:
                row = []
                for j in i:
                    if np.isnan(j):
                        row.append(255)
                    else:
                        row.append(j)
                channel.append(row)
            channels.append(channel)

        logger.info("Successfully colorized data into RGB channels")
        return channels  # [precip, solar, temp] -> [cyan, magenta, yellow]

    def plot_map_robinson(
        self,
        rgb_list: List[np.ndarray],
        title: str,
        save: bool = True,
        show: bool = False,
        projection: str = "PlateCarree",
        crop_lat: float = -60,
        dpi: int = 300,
        figsize: Tuple[int, int] = (16, 8),
    ) -> None:
        """
        Create and save a driver map using cartographic projection.

        Args:
            rgb_list: List of RGB arrays
            title: Plot title and filename
            save: Whether to save the plot
            show: Whether to display the plot
            projection: Map projection ('PlateCarree' or 'Robinson')
            crop_lat: Latitude below which to crop (hide Antarctica)
            dpi: Resolution for saved image
            figsize: Figure size tuple
        """
        # Validate input
        if len(rgb_list) != 3:
            raise ValueError("rgb_list must contain exactly 3 elements (R, G, B)")

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

        # Add cyclic column to handle antimeridian
        lon = np.linspace(-180.0, 180.0, rgb_data.shape[1], endpoint=False)
        rgb_cyc, lon_cyc = add_cyclic_point(rgb_data, coord=lon, axis=1)

        # Create alpha mask to hide regions below crop_lat
        H, Wc, _ = rgb_cyc.shape
        lats = np.linspace(90.0, -90.0, H)[:, None]
        alpha = np.repeat((lats >= crop_lat).astype(float), Wc, axis=1)

        # Plot the RGB data
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
            plt.title(title, fontsize=14, fontweight="bold")

        # Save plot
        if save and title:
            output_path = self.output_dir / f"{title}.png"
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
            plt.close(fig)

    def generate_pft_map(
        self,
        file_paths: List[str],
        pft_names: List[str],
        output_name: str,
        **plot_kwargs,
    ) -> None:
        """
        Generate a driver map for specific PFT data files.

        Args:
            file_paths: List of CSV file paths
            pft_names: List of PFT identifiers for filtering
            output_name: Name for the output plot
            **plot_kwargs: Additional arguments for plotting
        """
        dataframes = []

        for file_path, pft in zip(file_paths, pft_names):
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            df = self.load_csv_data(file_path, pft)
            if not df.empty:
                dataframes.append(df)

        if not dataframes:
            logger.error(f"No valid data found for {output_name}")
            return

        # Process data
        full_df = self.concatenate_dataframes(dataframes)
        rgb_list = self.colorize_data(full_df)

        # Generate plot
        self.plot_map_robinson(rgb_list, output_name, **plot_kwargs)
        logger.info(f"Generated map: {output_name}")


def create_file_mapping(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Create a mapping of available data files.

    Args:
        data_dir: Directory containing data files

    Returns:
        Dictionary mapping PFT and phase to file paths
    """
    data_path = Path(data_dir)
    file_mapping = {}

    # Define expected file patterns
    patterns = {
        "BET": {"SOS": "BET_*_SOS*.csv", "EOS": "BET_*_EOS*.csv"},
        "NET": {"SOS": "NET_*_SOS*.csv", "EOS": "NET_*_EOS*.csv"},
        "NDT": {"SOS": "NDT_*_SOS*.csv", "EOS": "NDT_*_EOS*.csv"},
        "SHR": {"SOS": "SHR_*_SOS*.csv", "EOS": "SHR_*_EOS*.csv"},
        "GRA": {"SOS": "GRA_*_SOS*.csv", "EOS": "GRA_*_EOS*.csv"},
        "BDT_5020": {"SOS": "BDT_50_20_*_SOS*.csv", "EOS": "BDT_50_20_*_EOS*.csv"},
        "BDT_2020": {"SOS": "BDT_-20_20_*_SOS*.csv", "EOS": "BDT_-20_20_*_EOS*.csv"},
        "BDT_2060": {"SOS": "BDT_-20_-60_*_SOS*.csv", "EOS": "BDT_-20_-60_*_EOS*.csv"},
        "BDT_5090": {"SOS": "BDT_50_90_*_SOS*.csv", "EOS": "BDT_50_90_*_EOS*.csv"},
    }

    # Find matching files
    for pft, phases in patterns.items():
        file_mapping[pft] = {}
        for phase, pattern in phases.items():
            matching_files = list(data_path.glob(pattern))
            if matching_files:
                # Use the most recent file if multiple matches
                file_mapping[pft][phase] = str(sorted(matching_files)[-1])

    return file_mapping


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate phenological driver maps")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing CSV data files",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save output plots"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots instead of just saving"
    )
    parser.add_argument(
        "--projection",
        choices=["PlateCarree", "Robinson"],
        default="PlateCarree",
        help="Map projection to use",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolution for saved images"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = DriverMapGenerator(args.data_dir, args.output_dir)

    # Create file mapping
    file_mapping = create_file_mapping(args.data_dir)

    # Plot settings
    plot_kwargs = {
        "show": args.show_plots,
        "projection": args.projection,
        "dpi": args.dpi,
        "crop_lat": -60,
    }

    # Generate individual PFT maps
    logger.info("Generating individual PFT maps...")

    for pft in ["BET", "NET", "NDT", "SHR", "GRA"]:
        for phase in ["SOS", "EOS"]:
            if pft in file_mapping and phase in file_mapping[pft]:
                file_path = file_mapping[pft][phase]
                output_name = f"{pft}_{phase}"

                generator.generate_pft_map(
                    [file_path], [pft], output_name, **plot_kwargs
                )

    # Generate combined BDT maps
    logger.info("Generating combined BDT maps...")

    bdt_types = ["BDT_5020", "BDT_2020", "BDT_2060", "BDT_5090"]

    for phase in ["SOS", "EOS"]:
        bdt_files = []
        bdt_names = []

        for bdt_type in bdt_types:
            if bdt_type in file_mapping and phase in file_mapping[bdt_type]:
                bdt_files.append(file_mapping[bdt_type][phase])
                bdt_names.append(bdt_type)

        if bdt_files:
            output_name = f"BDT_Combined_{phase}"
            generator.generate_pft_map(bdt_files, bdt_names, output_name, **plot_kwargs)

    # Generate comprehensive maps
    logger.info("Generating comprehensive maps...")

    for phase in ["SOS", "EOS"]:
        all_files = []
        all_names = []

        for pft in file_mapping:
            if phase in file_mapping[pft]:
                all_files.append(file_mapping[pft][phase])
                all_names.append(pft)

        if all_files:
            output_name = f"All_PFTs_{phase}"
            generator.generate_pft_map(all_files, all_names, output_name, **plot_kwargs)

    logger.info("Driver map generation completed successfully!")


if __name__ == "__main__":
    main()
