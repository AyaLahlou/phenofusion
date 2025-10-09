"""
Example Script: Heatmap Analysis Demonstrations

This script demonstrates various ways to use the heatmap analysis module
for phenological feature importance and attention pattern visualization.

Run this script to generate example analyses for different cities and conditions.

Author: Phenofusion Team
Date: October 2025
"""

import sys
import logging
from pathlib import Path

# Add phenofusion to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from phenofusion.analysis.heatmap_analysis import (
    HeatmapAnalyzer,
    analyze_city_heatmaps,
    create_analysis_dataframe,
    GeographicLocalizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_basic_city_analysis():
    """Example 1: Basic city analysis with default parameters."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic City Analysis")
    print("=" * 60)

    try:
        # Analyze Budapest for start of growing season
        logger.info("Analyzing Budapest (BDT, start of season)")
        analyze_city_heatmaps(city="Budapest", pft="BDT", season="sos")
        logger.info("Budapest analysis completed successfully!")

    except Exception as e:
        logger.error(f"Example 1 failed: {e}")


def example_2_multi_city_comparison():
    """Example 2: Compare multiple cities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-City Comparison")
    print("=" * 60)

    cities = [
        {"city": "Moscow", "pft": "BDT", "season": "sos"},
        {"city": "Tokyo", "pft": "BDT", "season": "eos"},
        {"city": "New York", "pft": "BDT", "season": "sos"},
    ]

    for i, params in enumerate(cities, 1):
        try:
            logger.info(f"Analyzing city {i}/{len(cities)}: {params['city']}")
            analyze_city_heatmaps(**params)
            logger.info(f"{params['city']} analysis completed!")

        except Exception as e:
            logger.error(f"Failed to analyze {params['city']}: {e}")


def example_3_specific_year_analysis():
    """Example 3: Analysis for a specific year."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Specific Year Analysis")
    print("=" * 60)

    try:
        # Analyze Berlin for year 2006
        logger.info("Analyzing Berlin for year 2006")
        analyze_city_heatmaps(
            city="Berlin", pft="BDT", season="sos", specific_year=2006
        )
        logger.info("Berlin 2006 analysis completed!")

    except Exception as e:
        logger.error(f"Example 3 failed: {e}")


def example_4_different_plant_types():
    """Example 4: Different plant functional types."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Different Plant Functional Types")
    print("=" * 60)

    # Note: This example assumes you have data for different PFTs
    plant_types = [
        {"city": "Vancouver", "pft": "BDT", "season": "sos"},
        {"city": "Amazon", "pft": "BET", "season": "sos"},  # If BET data available
    ]

    for params in plant_types:
        try:
            logger.info(f"Analyzing {params['city']} with PFT {params['pft']}")
            analyze_city_heatmaps(**params)
            logger.info(f"Analysis completed for {params['city']} ({params['pft']})")

        except Exception as e:
            logger.error(f"Failed to analyze {params['city']} ({params['pft']}): {e}")


def example_5_custom_analyzer():
    """Example 5: Using HeatmapAnalyzer directly with custom settings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Analyzer Usage")
    print("=" * 60)

    try:
        # Create custom output directory
        custom_output_dir = "custom_heatmap_plots"

        # Initialize analyzer with custom settings
        analyzer = HeatmapAnalyzer(output_dir=custom_output_dir)
        logger.info(f"Created analyzer with output directory: {custom_output_dir}")

        # Example file paths (modify these to match your data)
        data_path = "/burg/glab/users/al4385/data/TFT_30/BDT_50_20.pickle"
        pred_path = "/burg/glab/users/al4385/predictions/TFT_30_0429/pred_BDT_50_20.pkl"
        coord_path = "/burg/glab/users/al4385/data/coordinates/BDT_50_20.parquet"

        # Check if files exist
        if not all(Path(p).exists() for p in [data_path, pred_path, coord_path]):
            logger.warning("Data files not found, skipping custom analyzer example")
            return

        # Create analysis DataFrame
        df = create_analysis_dataframe(data_path, pred_path, coord_path)
        logger.info(f"Created DataFrame with {len(df)} records")

        # Filter for a specific region (e.g., around Paris)
        lat_range, lon_range = GeographicLocalizer.get_city_bounding_box("Paris")
        df_filtered = GeographicLocalizer.localize_df(df, lat_range, lon_range)
        logger.info(f"Filtered to {len(df_filtered)} records for Paris region")

        # Generate visualizations
        analyzer.generate_feature_importance_heatmap(
            df_filtered, pred_path, data_path, season="sos", title="Paris_Custom"
        )
        analyzer.generate_attention_heatmap(
            df_filtered, pred_path, season="sos", title="Paris_Custom"
        )

        logger.info("Custom analyzer example completed!")

    except Exception as e:
        logger.error(f"Example 5 failed: {e}")


def example_6_geographic_analysis():
    """Example 6: Geographic analysis with coordinate inspection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Geographic Analysis")
    print("=" * 60)

    try:
        # Demonstrate geographic utilities
        cities = ["London", "Sydney", "Cairo", "São Paulo"]

        for city in cities:
            try:
                lat_range, lon_range = GeographicLocalizer.get_city_bounding_box(city)
                lat_interval = GeographicLocalizer.check_latitude_interval(lat_range)

                logger.info(f"{city}:")
                logger.info(f"  Latitude range: {lat_range}")
                logger.info(f"  Longitude range: {lon_range}")
                logger.info(f"  Latitude interval: {lat_interval}")

            except Exception as e:
                logger.error(f"Failed to process {city}: {e}")

    except Exception as e:
        logger.error(f"Example 6 failed: {e}")


def run_all_examples():
    """Run all example demonstrations."""
    print("🌿 PHENOFUSION HEATMAP ANALYSIS EXAMPLES 🌿")
    print(
        "This script demonstrates various capabilities of the heatmap analysis module"
    )
    print(
        "\nNote: Some examples may fail if the required data files are not available."
    )
    print("This is expected and demonstrates the error handling capabilities.")

    examples = [
        example_1_basic_city_analysis,
        example_2_multi_city_comparison,
        example_3_specific_year_analysis,
        example_4_different_plant_types,
        example_5_custom_analyzer,
        example_6_geographic_analysis,
    ]

    successful_examples = 0

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
            successful_examples += 1
        except Exception as e:
            logger.error(f"Example {i} encountered an error: {e}")

    print("\n" + "=" * 60)
    print("EXAMPLES SUMMARY")
    print("=" * 60)
    print(f"Successfully completed: {successful_examples}/{len(examples)} examples")
    print("\nGenerated plots can be found in:")
    print("- feature_importance_plots/ (default output directory)")
    print("- custom_heatmap_plots/ (if Example 5 ran successfully)")
    print("\nFor more information, see docs/heatmap_analysis.md")


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        example_number = sys.argv[1]

        examples_map = {
            "1": example_1_basic_city_analysis,
            "2": example_2_multi_city_comparison,
            "3": example_3_specific_year_analysis,
            "4": example_4_different_plant_types,
            "5": example_5_custom_analyzer,
            "6": example_6_geographic_analysis,
        }

        if example_number in examples_map:
            print(f"Running Example {example_number}")
            examples_map[example_number]()
        else:
            print(f"Example {example_number} not found. Available: 1-6")
            print("Usage: python examples_heatmap_analysis.py [1-6]")
            print("Or run without arguments to run all examples")
    else:
        # Run all examples
        run_all_examples()
