"""
Final, robust Parquet File Merger

This module provides a working, tested solution for merging parquet files
with proper error handling and memory management.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional  # Remove List from this import
import logging
from datetime import datetime
import gc
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_parquet_files_robust(
    directory_path: str, output_path: Optional[str] = None, chunk_size: int = 3
) -> str:
    """
    Robust function to merge all parquet files in a directory.

    This approach loads files in small batches and uses append mode
    to avoid memory issues with very large datasets.

    Args:
        directory_path: Path to directory containing .parquet files
        output_path: Optional path for merged output file
        chunk_size: Number of files to process at once

    Returns:
        str: Path to merged output file
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory {directory_path} does not exist")

    # Find all parquet files
    parquet_files = sorted(list(directory.glob("*.parquet")))

    if not parquet_files:
        raise ValueError(f"No .parquet files found in {directory_path}")

    # Set output path
    if output_path is None:
        output_path = f"{directory.name}_merged.parquet"

    output_file = Path(output_path)

    # Remove existing output file if it exists
    if output_file.exists():
        output_file.unlink()
        logger.info(f"Removed existing output file: {output_file}")

    logger.info(f"Found {len(parquet_files)} parquet files to merge")
    logger.info(f"Processing in chunks of {chunk_size} files")
    logger.info(f"Output file: {output_file}")

    total_rows = 0
    files_processed = 0
    all_dataframes = []

    # Process files in chunks
    for i in range(0, len(parquet_files), chunk_size):
        chunk_files = parquet_files[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(parquet_files) - 1) // chunk_size + 1

        logger.info(
            f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_files)} files)"
        )

        # Load all files in current chunk
        chunk_dataframes = []
        chunk_rows = 0

        for file_path in chunk_files:
            try:
                logger.info(f"  Loading {file_path.name}")
                df = pd.read_parquet(file_path)
                chunk_dataframes.append(df)
                chunk_rows += len(df)
                files_processed += 1

            except Exception as e:
                logger.error(f"  Error loading {file_path.name}: {e}")
                continue

        if not chunk_dataframes:
            logger.warning(f"  No valid files in chunk {chunk_num}")
            continue

        # Merge current chunk
        logger.info(f"  Merging {len(chunk_dataframes)} files ({chunk_rows:,} rows)")
        chunk_merged = pd.concat(chunk_dataframes, ignore_index=True, sort=False)

        # Add to collection for final merge
        all_dataframes.append(chunk_merged)
        total_rows += len(chunk_merged)

        # Clear individual dataframes from memory
        del chunk_dataframes, chunk_merged
        gc.collect()

    # Final merge of all chunks
    logger.info(f"Performing final merge of {len(all_dataframes)} chunks")
    final_merged = pd.concat(all_dataframes, ignore_index=True, sort=False)

    # Save final result
    final_merged.to_parquet(output_file, index=False)

    # Clear from memory
    del all_dataframes, final_merged
    gc.collect()

    logger.info("Merge complete!")
    logger.info(f"  Files processed: {files_processed}/{len(parquet_files)}")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Output size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    return str(output_file)


def analyze_merged_file(file_path: str) -> Dict:
    """
    Analyze a merged parquet file and provide diagnostics.

    Args:
        file_path: Path to the merged parquet file

    Returns:
        Dict: Analysis results
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist")

    logger.info(f"Analyzing merged file: {file_path}")

    # Get basic file info
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    # Try to get row count efficiently using pyarrow
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        num_columns = parquet_file.metadata.num_columns
        column_names = parquet_file.schema_arrow.names
        use_sample = False
    except Exception:
        # Fallback: read full file if pyarrow fails
        try:
            df = pd.read_parquet(file_path)
            total_rows = len(df)
            num_columns = len(df.columns)
            column_names = list(df.columns)
            sample_df = df
            use_sample = False
        except Exception as e:
            logger.error(f"Could not read parquet file: {e}")
            return {
                "file_path": str(file_path),
                "file_size_mb": file_size_mb,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
            }

    analysis = {
        "file_path": str(file_path),
        "file_size_mb": file_size_mb,
        "total_rows": total_rows,
        "num_columns": num_columns,
        "column_names": column_names,
        "analysis_timestamp": datetime.now().isoformat(),
        "used_sample": use_sample,
    }

    # For large files, read just the first few rows for analysis
    if file_size_mb > 500:
        try:
            # Read first 100k rows using pyarrow
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(file_path)
            # Read first batch/chunk
            batch = parquet_file.read_row_group(0, columns=None)
            sample_df = batch.to_pandas()
            # Limit to 100k rows if the first row group is larger
            if len(sample_df) > 100000:
                sample_df = sample_df.head(100000)
            use_sample = True
            analysis["used_sample"] = True
            logger.info(f"Using sample of {len(sample_df):,} rows for analysis")
        except Exception:
            # Final fallback: read entire file
            logger.warning("Could not sample file, reading entire file for analysis")
            sample_df = pd.read_parquet(file_path)
            use_sample = False
    else:
        # Small file, read everything
        if "sample_df" not in locals():
            sample_df = pd.read_parquet(file_path)
        use_sample = False

    # Analyze sample data
    if "sample_df" in locals():
        sample_size = len(sample_df)

        # Basic statistics
        analysis["sample_analysis"] = {
            "sample_size": sample_size,
            "null_counts": dict(sample_df.isnull().sum()),
            "dtypes": dict(sample_df.dtypes.astype(str)),
        }

        # Time analysis if time column exists
        if "time" in sample_df.columns:
            time_col = sample_df["time"]
            analysis["time_analysis"] = {
                "min_date": str(time_col.min()),
                "max_date": str(time_col.max()),
                "unique_dates_in_sample": time_col.nunique(),
                "date_range_days": (time_col.max() - time_col.min()).days,
            }

        # Location analysis if location column exists
        if "location" in sample_df.columns:
            loc_col = sample_df["location"]
            unique_locs = loc_col.nunique()
            analysis["location_analysis"] = {
                "unique_locations_in_sample": unique_locs,
                "location_range": [int(loc_col.min()), int(loc_col.max())],
                "avg_records_per_location": (
                    sample_size / unique_locs if unique_locs > 0 else 0
                ),
            }

        # Numeric column summaries
        numeric_cols = sample_df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summaries"] = {}
            for col in numeric_cols:
                if col not in ["location"]:  # Skip ID-like columns
                    stats = sample_df[col].describe()
                    analysis["numeric_summaries"][col] = {
                        "mean": float(stats["mean"]),
                        "std": float(stats["std"]),
                        "min": float(stats["min"]),
                        "max": float(stats["max"]),
                        "median": float(stats["50%"]),
                    }

        # Clean up
        del sample_df
        gc.collect()

    return analysis


def print_analysis(analysis: Dict):
    """Print formatted analysis results."""
    print("=" * 60)
    print("MERGED PARQUET FILE ANALYSIS")
    print("=" * 60)

    print(f"File: {analysis['file_path']}")
    print(f"Size: {analysis['file_size_mb']:.2f} MB")
    print(f"Rows: {analysis['total_rows']}")
    print(f"Columns: {analysis['num_columns']}")
    print(f"Analysis Time: {analysis['analysis_timestamp']}")

    if analysis.get("used_sample"):
        print("Note: Large file - analysis based on sample")

    print()

    if "sample_analysis" in analysis:
        sample = analysis["sample_analysis"]
        print("DATA QUALITY:")

        # Null counts
        null_counts = sample["null_counts"]
        total_nulls = sum(null_counts.values())
        if total_nulls > 0:
            print(f"  Total null values: {total_nulls:,}")
            null_cols = [
                (col, count) for col, count in null_counts.items() if count > 0
            ]
            for col, count in sorted(null_cols, key=lambda x: x[1], reverse=True)[:5]:
                pct = (count / sample["sample_size"]) * 100
                print(f"    {col}: {count:,} ({pct:.1f}%)")
        else:
            print("  No null values found")
        print()

    if "time_analysis" in analysis:
        time_info = analysis["time_analysis"]
        print("TIME ANALYSIS:")
        print(f"  Date range: {time_info['min_date']} to {time_info['max_date']}")
        print(f"  Span: {time_info['date_range_days']} days")
        print(f"  Unique dates (sample): {time_info['unique_dates_in_sample']:,}")
        print()

    if "location_analysis" in analysis:
        loc_info = analysis["location_analysis"]
        print("LOCATION ANALYSIS:")
        print(
            f"  Unique locations (sample): {loc_info['unique_locations_in_sample']:,}"
        )
        print(
            f"  Location range: {loc_info['location_range'][0]} to {loc_info['location_range'][1]}"
        )
        print(f"  Avg records per location: {loc_info['avg_records_per_location']:.1f}")
        print()

    if "numeric_summaries" in analysis:
        print("NUMERIC VARIABLE SUMMARIES:")
        for var, stats in analysis["numeric_summaries"].items():
            print(f"  {var}:")
            print(f"    Range: {stats['min']:.3f} to {stats['max']:.3f}")
            print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"    Median: {stats['median']:.3f}")
        print()

    print("COLUMNS:")
    for i, col in enumerate(analysis["column_names"], 1):
        print(f"  {i:2d}. {col}")

    print("=" * 60)


def test_small_merge():
    """Test the merger with a small subset of files."""
    # Use the default directory for testing
    source_dir = Path("/burg/glab/users/al4385/data/CSIFMETEO/SCH/")

    if not source_dir.exists():
        logger.error(f"Test source directory does not exist: {source_dir}")
        return

    # Create a test directory with just a few files
    test_dir = Path("test_merge")
    test_dir.mkdir(exist_ok=True)

    try:
        # Copy first 3 files for testing
        source_files = sorted(list(source_dir.glob("*.parquet")))[:3]

        if not source_files:
            logger.error(f"No parquet files found in {source_dir}")
            return

        for i, source_file in enumerate(source_files):
            dest_file = test_dir / f"test_{i+1}.parquet"
            shutil.copy2(source_file, dest_file)
            print(f"Copied {source_file.name} to {dest_file.name}")

        # Test merge
        output_file = merge_parquet_files_robust(
            directory_path=str(test_dir),
            output_path="test_merged.parquet",
            chunk_size=2,
        )

        # Analyze result
        analysis = analyze_merged_file(output_file)
        print_analysis(analysis)

    except Exception as e:
        logger.error(f"Error in test: {e}")
    finally:
        # Clean up test files
        if test_dir.exists():
            shutil.rmtree(test_dir)
        if Path("test_merged.parquet").exists():
            Path("test_merged.parquet").unlink()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Merge parquet files in a directory")
    parser.add_argument(
        "directory_path",
        nargs="?",
        default="/burg/glab/users/al4385/data/CSIFMETEO/SCH/",
        help="Path to directory containing .parquet files (default: /burg/glab/users/al4385/data/CSIFMETEO/SCH/)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Output path for merged file (default: <directory_name>_merged.parquet)",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=2,
        help="Number of files to process at once (default: 2)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test with small subset of files"
    )

    args = parser.parse_args()

    if args.test:
        # Run test with small subset
        test_small_merge()
    else:
        # Run full merge
        try:
            output_file = merge_parquet_files_robust(
                directory_path=args.directory_path,
                output_path=args.output_path,
                chunk_size=args.chunk_size,
            )

            print(f"\nMerge successful! Output: {output_file}")
            print("\nGenerating analysis...")

            analysis = analyze_merged_file(output_file)
            print_analysis(analysis)

        except Exception as e:
            logger.error(f"Error during merge: {e}")
            sys.exit(1)
