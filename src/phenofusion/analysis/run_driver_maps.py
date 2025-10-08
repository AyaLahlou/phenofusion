#!/usr/bin/env python3
"""
Simple runner script for generating driver maps.

This script provides a simpler interface to the driver map generator,
automatically detecting available data files and generating all possible maps.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the module
sys.path.append(str(Path(__file__).parent))

from generate_driver_maps import DriverMapGenerator, create_file_mapping
import logging


def run_driver_map_generation(
    data_dir: str = "/burg/home/al4385/code/phenology_analysis/drivers_data/",
    output_dir: str = "./driver_maps_output/",
    show_plots: bool = False,
    projection: str = "PlateCarree",
):
    """
    Run the driver map generation with default settings.

    Args:
        data_dir: Directory containing CSV data files
        output_dir: Directory to save output plots
        show_plots: Whether to display plots
        projection: Map projection to use
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting driver map generation...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return False

    # Initialize generator
    generator = DriverMapGenerator(data_dir, output_dir)

    # Create file mapping
    file_mapping = create_file_mapping(data_dir)

    if not file_mapping:
        logger.error("No valid data files found!")
        return False

    # Log available files
    logger.info("Found data files:")
    for pft, phases in file_mapping.items():
        for phase, filepath in phases.items():
            logger.info(f"  {pft} {phase}: {os.path.basename(filepath)}")

    # Plot settings
    plot_kwargs = {
        "show": show_plots,
        "projection": projection,
        "dpi": 300,
        "crop_lat": -60,
    }

    success_count = 0
    total_count = 0

    # Generate individual PFT maps
    logger.info("\n=== Generating individual PFT maps ===")

    for pft in ["BET", "NET", "NDT"]:
        for phase in ["SOS", "EOS"]:
            total_count += 1
            if pft in file_mapping and phase in file_mapping[pft]:
                try:
                    file_path = file_mapping[pft][phase]
                    output_name = f"{pft}_{phase}"

                    generator.generate_pft_map(
                        [file_path], [pft], output_name, **plot_kwargs
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to generate {pft}_{phase}: {e}")
            else:
                logger.warning(f"No data file found for {pft}_{phase}")

    # Generate combined BDT maps
    logger.info("\n=== Generating combined BDT maps ===")

    bdt_types = ["BDT_5020", "BDT_2020", "BDT_2060", "BDT_5090"]

    for phase in ["SOS", "EOS"]:
        total_count += 1
        bdt_files = []
        bdt_names = []

        for bdt_type in bdt_types:
            if bdt_type in file_mapping and phase in file_mapping[bdt_type]:
                bdt_files.append(file_mapping[bdt_type][phase])
                bdt_names.append(bdt_type)

        if bdt_files:
            try:
                output_name = f"BDT_Combined_{phase}"
                generator.generate_pft_map(
                    bdt_files, bdt_names, output_name, **plot_kwargs
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to generate BDT_Combined_{phase}: {e}")
        else:
            logger.warning(f"No BDT data files found for {phase}")

    # Generate comprehensive maps
    logger.info("\n=== Generating comprehensive maps ===")

    for phase in ["SOS", "EOS"]:
        total_count += 1
        all_files = []
        all_names = []

        for pft in file_mapping:
            if phase in file_mapping[pft]:
                all_files.append(file_mapping[pft][phase])
                all_names.append(pft)

        if all_files:
            try:
                output_name = f"All_PFTs_{phase}"
                generator.generate_pft_map(
                    all_files, all_names, output_name, **plot_kwargs
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to generate All_PFTs_{phase}: {e}")
        else:
            logger.warning(f"No data files found for comprehensive {phase} map")

    logger.info("\n=== Generation Complete ===")
    logger.info(f"Successfully generated {success_count}/{total_count} maps")
    logger.info(f"Output saved to: {output_dir}")

    return success_count > 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate phenological driver maps (simple runner)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/burg/home/al4385/code/phenology_analysis/drivers_data/",
        help="Directory containing CSV data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./driver_maps_output/",
        help="Directory to save output plots",
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

    args = parser.parse_args()

    success = run_driver_map_generation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        projection=args.projection,
    )

    if success:
        print("\n✅ Driver map generation completed successfully!")
    else:
        print("\n❌ Driver map generation failed!")
        sys.exit(1)
