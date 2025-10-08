#!/usr/bin/env python3
"""
Simple runner script for generating sensitivity analysis plots.

This script provides a simplified interface to the sensitivity analyzer,
automatically processing temporal changes in climate driver sensitivity
between different years.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the module
sys.path.append(str(Path(__file__).parent))

from generate_sensitivity_plots import (
    AnalysisConfig,
    PhenologicalDataProcessor,
    SensitivityAnalyzer,
)
import logging


def run_sensitivity_analysis(
    data_dir: str = "/burg/glab/users/al4385/data/TFT_30_40years/",
    pred_dir: str = "/burg/glab/users/al4385/predictions/TFT_30_40years/",
    coord_dir: str = "/burg/glab/users/al4385/data/coordinates/",
    output_dir: str = "./sensitivity_analysis_output/",
    year1: int = 1985,
    year2: int = 2020,
    seasons: list = ["sos", "eos"],
    show_plots: bool = False,
):
    """
    Run sensitivity analysis with default settings.

    Args:
        data_dir: Directory containing data pickle files
        pred_dir: Directory containing prediction pickle files
        coord_dir: Directory containing coordinate parquet files
        output_dir: Directory to save output plots
        year1: First year for comparison (baseline)
        year2: Second year for comparison
        seasons: List of seasons to analyze
        show_plots: Whether to display plots
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting sensitivity analysis...")
    logger.info(f"Comparing {year1} vs {year2}")
    logger.info(f"Seasons: {', '.join(seasons)}")
    logger.info(f"Output directory: {output_dir}")

    # Check if directories exist
    for dir_path, name in [
        (data_dir, "Data"),
        (pred_dir, "Prediction"),
        (coord_dir, "Coordinate"),
    ]:
        if not os.path.exists(dir_path):
            logger.error(f"{name} directory does not exist: {dir_path}")
            return False

    # Create configuration
    config = AnalysisConfig(
        data_directory=data_dir,
        pred_directory=pred_dir,
        coord_directory=coord_dir,
        output_directory=output_dir,
        years=[year1, year2],
        seasons=seasons,
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
    try:
        processor = PhenologicalDataProcessor(config)
        analyzer = SensitivityAnalyzer(processor)
        logger.info("Successfully initialized analysis components")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

    success_count = 0
    total_plots = len(seasons) * 3  # 3 drivers per season

    # Process each season
    for season in seasons:
        logger.info(f"\n=== Processing {season.upper()} season ===")

        try:
            # Calculate temporal differences
            temp_diff, solar_diff, precip_diff = (
                analyzer.calculate_temporal_differences(year1, year2, season)
            )

            if temp_diff.size == 0:
                logger.error(f"No data available for {season} season")
                continue

            # Generate plots for each driver
            drivers = [
                (temp_diff, "temperature"),
                (solar_diff, "solar_radiation"),
                (precip_diff, "water_availability"),
            ]

            for diff_data, driver_name in drivers:
                try:
                    analyzer.plot_difference_map(
                        diff_data,
                        driver_name,
                        season,
                        year1,
                        year2,
                        save=True,
                        show=show_plots,
                        dpi=300,
                    )
                    success_count += 1
                    logger.info(f"✅ Generated {driver_name} plot for {season}")
                except Exception as e:
                    logger.error(
                        f"❌ Failed to generate {driver_name} plot for {season}: {e}"
                    )

        except Exception as e:
            logger.error(f"❌ Failed to process {season} season: {e}")
            continue

    logger.info("\n=== Analysis Complete ===")
    logger.info(f"Successfully generated {success_count}/{total_plots} plots")
    logger.info(f"Output saved to: {output_dir}")

    return success_count > 0


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis plots (simple runner)"
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

    args = parser.parse_args()

    success = run_sensitivity_analysis(
        data_dir=args.data_dir,
        pred_dir=args.pred_dir,
        coord_dir=args.coord_dir,
        output_dir=args.output_dir,
        year1=args.year1,
        year2=args.year2,
        seasons=args.seasons,
        show_plots=args.show_plots,
    )

    if success:
        print("\n✅ Sensitivity analysis completed successfully!")
        print(
            "📊 Generated temporal difference maps showing changes in climate driver sensitivity"
        )
    else:
        print("\n❌ Sensitivity analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
