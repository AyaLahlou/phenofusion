"""
Heatmap Analysis Runner Script

Simple command-line interface for generating phenological heatmap analyses.
This script provides an easy way to run heatmap analysis for different cities,
plant functional types, and seasons.

Usage:
    python run_heatmap_analysis.py --city "Budapest" --pft "BDT" --season "sos"
    python run_heatmap_analysis.py --city "Tokyo" --pft "BET" --season "eos" --year 2006

Author: Phenofusion Team
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
import logging

# Add the phenofusion package to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from phenofusion.analysis.heatmap_analysis import analyze_city_heatmaps

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run heatmap analysis from command line."""
    parser = argparse.ArgumentParser(
        description="Generate phenological heatmap analysis for cities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --city "Budapest" --pft "BDT" --season "sos"
  %(prog)s --city "Tokyo" --pft "BET" --season "eos" --year 2006
  %(prog)s --city "New York" --season "sos" --output-dir "./plots"
        """,
    )

    # Required arguments
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        help="Name of the city to analyze (e.g., 'Budapest', 'Tokyo', 'New York')",
    )

    # Optional arguments
    parser.add_argument(
        "--pft",
        type=str,
        default="BDT",
        choices=["BDT", "BET", "NDT", "NET", "GRA"],
        help="Plant functional type (default: BDT)",
    )

    parser.add_argument(
        "--season",
        type=str,
        default="sos",
        choices=["sos", "eos"],
        help="Season type: 'sos' (start of season) or 'eos' (end of season) (default: sos)",
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Specific year to analyze (optional, if not provided analyzes all available years)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="feature_importance_plots",
        help="Output directory for plots (default: feature_importance_plots)",
    )

    # Data directory arguments (for custom installations)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/burg/glab/users/al4385/data/TFT_30",
        help="Base directory for data files",
    )

    parser.add_argument(
        "--pred-dir",
        type=str,
        default="/burg/glab/users/al4385/predictions/TFT_30_0429",
        help="Base directory for prediction files",
    )

    parser.add_argument(
        "--coord-dir",
        type=str,
        default="/burg/glab/users/al4385/data/coordinates",
        help="Base directory for coordinate files",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate year
    if args.year and (args.year < 2000 or args.year > 2020):
        logger.error("Year must be between 2000 and 2020")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Starting heatmap analysis for {args.city}")
    logger.info(
        f"Parameters: PFT={args.pft}, Season={args.season}, Year={args.year or 'all'}"
    )

    try:
        # Import here to avoid circular imports
        from phenofusion.analysis.heatmap_analysis import HeatmapAnalyzer

        # Set output directory for analyzer
        original_analyzer = HeatmapAnalyzer
        HeatmapAnalyzer.__init__ = (
            lambda self, output_dir=args.output_dir: original_analyzer.__init__(
                self, output_dir
            )
        )

        # Run analysis
        analyze_city_heatmaps(
            city=args.city,
            pft=args.pft,
            season=args.season,
            specific_year=args.year,
            base_data_dir=args.data_dir,
            base_pred_dir=args.pred_dir,
            base_coord_dir=args.coord_dir,
        )

        logger.info(
            f"Analysis completed successfully! Plots saved to {args.output_dir}"
        )

    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        logger.error("Please check that all data directories and files exist")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def run_example_analyses():
    """Run a set of example analyses for different cities and conditions."""
    examples = [
        {"city": "Budapest", "pft": "BDT", "season": "sos"},
        {"city": "Tokyo", "pft": "BDT", "season": "eos"},
        {"city": "New York", "pft": "BDT", "season": "sos", "year": 2006},
        {"city": "Moscow", "pft": "BDT", "season": "eos"},
    ]

    logger.info("Running example analyses...")

    for i, params in enumerate(examples, 1):
        logger.info(f"Example {i}/{len(examples)}: {params}")
        try:
            analyze_city_heatmaps(**params)
            logger.info(f"Example {i} completed successfully")
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")


if __name__ == "__main__":
    main()
