#!/usr/bin/env python3
"""
Diagnostic Script for BET Driver Data

This script helps diagnose why you're getting NaNs in your BET driver maps
by analyzing the data, predictions, and phenology detection.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse


def analyze_data_file(data_path):
    """Analyze the structure and contents of the data file."""
    print("\n" + "=" * 60)
    print("DATA FILE ANALYSIS")
    print("=" * 60)

    with open(data_path, "rb") as fp:
        data = pickle.load(fp)

    # Analyze test set
    test_data = data["data_sets"]["test"]

    print("\nTest set structure:")
    for key, value in test_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    # Analyze IDs
    sample_ids = test_data["id"]
    print("\nSample IDs:")
    print(f"  Total samples: {len(sample_ids)}")
    print(f"  First 5: {sample_ids[:5]}")

    # Extract locations
    locations = [int(id_str[0].split("_")[0]) for id_str in sample_ids]
    unique_locations = len(set(locations))
    print(f"  Unique locations: {unique_locations}")

    # Analyze targets (CSIF)
    targets = test_data["target"]
    print("\nTarget (CSIF) statistics:")
    print(f"  Shape: {targets.shape}")
    print(f"  Min: {targets.min():.4f}")
    print(f"  Max: {targets.max():.4f}")
    print(f"  Mean: {targets.mean():.4f}")
    print(f"  Std: {targets.std():.4f}")

    return data


def analyze_predictions(pred_path):
    """Analyze the predictions file."""
    print("\n" + "=" * 60)
    print("PREDICTIONS FILE ANALYSIS")
    print("=" * 60)

    with open(pred_path, "rb") as fp:
        preds = pickle.load(fp)

    print("\nPrediction structure:")
    for key, value in preds.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    # Analyze attention scores
    if "attention_scores" in preds:
        att_scores = preds["attention_scores"]
        print("\nAttention scores:")
        print(f"  Shape: {att_scores.shape}")
        print(f"  Min: {att_scores.min():.4f}")
        print(f"  Max: {att_scores.max():.4f}")
        print(f"  Mean: {att_scores.mean():.4f}")

    # Analyze historical selection weights
    if "historical_selection_weights" in preds:
        hist_weights = preds["historical_selection_weights"]
        print("\nHistorical selection weights:")
        print(f"  Shape: {hist_weights.shape}")
        print(
            f"  Features: {hist_weights.shape[2] if len(hist_weights.shape) > 2 else 'N/A'}"
        )
        print(f"  Min: {hist_weights.min():.4f}")
        print(f"  Max: {hist_weights.max():.4f}")

        # Check individual features
        if len(hist_weights.shape) == 3:
            feature_names = ["CSIF", "Tmin", "Tmax", "Rad", "Precip", "Photo", "SM"]
            print("\n  Feature statistics:")
            for i, name in enumerate(feature_names[: hist_weights.shape[2]]):
                feat_data = hist_weights[:, :, i]
                print(
                    f"    {name}: min={feat_data.min():.4f}, max={feat_data.max():.4f}, mean={feat_data.mean():.4f}"
                )

    return preds


def analyze_coordinates(coord_path):
    """Analyze the coordinates file."""
    print("\n" + "=" * 60)
    print("COORDINATES FILE ANALYSIS")
    print("=" * 60)

    coords = pd.read_parquet(coord_path)

    print("\nCoordinate data:")
    print(f"  Total locations: {len(coords)}")
    print(f"  Unique locations: {coords['location'].nunique()}")
    print(f"  Columns: {list(coords.columns)}")

    print(
        f"\nLatitude range: {coords['latitude'].min():.2f} to {coords['latitude'].max():.2f}"
    )
    print(
        f"Longitude range: {coords['longitude'].min():.2f} to {coords['longitude'].max():.2f}"
    )

    # Show distribution
    lat_bins = pd.cut(coords["latitude"], bins=10)
    print("\nLatitude distribution:")
    print(lat_bins.value_counts().sort_index())

    return coords


def analyze_phenology_detection(
    data, preds, coords, forecast_window=30, min_diff=0.05, min_slope=0.001
):
    """Analyze phenology detection and see where samples are identified."""
    print("\n" + "=" * 60)
    print("PHENOLOGY DETECTION ANALYSIS")
    print("=" * 60)

    # Create analysis DataFrame
    test_data = data["data_sets"]["test"]

    df = pd.DataFrame(
        {
            "Index": test_data["id"].flatten(),
            "CSIF": test_data["target"].flatten(),
        }
    )

    df[["location", "time"]] = df["Index"].str.split("_", n=1, expand=True)
    df["location"] = df["location"].astype(int)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by=["location", "time"])
    df["doy"] = df["time"].dt.dayofyear
    df = df.drop(columns=["Index"])
    df = pd.merge(coords, df, on="location", how="left")

    print("\nAnalysis DataFrame created:")
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

    # Analyze batches
    batch_size = forecast_window
    total_batches = len(df) // batch_size

    print("\nBatch analysis:")
    print(f"  Batch size: {batch_size}")
    print(f"  Total possible batches: {total_batches}")

    # Detect phenology
    SOS_indices = []
    EOS_indices = []

    slope_distribution = []
    csif_range_distribution = []

    sos_by_lat = []
    eos_by_lat = []

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]

        if len(batch_df) < batch_size:
            continue

        if batch_df["location"].nunique() > 1:
            continue

        csif_values = batch_df["CSIF"].values
        csif_range = abs(csif_values[-1] - csif_values[0])
        csif_range_distribution.append(csif_range)

        if csif_range < min_diff:
            continue

        x = np.arange(len(csif_values))
        slope, _, _, _, _ = linregress(x, csif_values)
        slope_distribution.append(slope)

        if slope >= min_slope:
            SOS_indices.append(start)
            lat = batch_df["latitude"].iloc[0]
            sos_by_lat.append(lat)
        elif slope <= -min_slope - 0.0005:
            EOS_indices.append(start)
            lat = batch_df["latitude"].iloc[0]
            eos_by_lat.append(lat)

    print("\nDetection results:")
    print(f"  SOS samples detected: {len(SOS_indices)}")
    print(f"  EOS samples detected: {len(EOS_indices)}")
    print(
        f"  Detection rate: {100*(len(SOS_indices)+len(EOS_indices))/total_batches:.2f}%"
    )

    # Analyze distributions
    if slope_distribution:
        print("\nSlope distribution (samples with sufficient CSIF range):")
        print(f"  Min: {np.min(slope_distribution):.6f}")
        print(f"  Max: {np.max(slope_distribution):.6f}")
        print(f"  Mean: {np.mean(slope_distribution):.6f}")
        print(f"  Median: {np.median(slope_distribution):.6f}")

    if csif_range_distribution:
        print("\nCSIF range distribution:")
        print(f"  Min: {np.min(csif_range_distribution):.6f}")
        print(f"  Max: {np.max(csif_range_distribution):.6f}")
        print(f"  Mean: {np.mean(csif_range_distribution):.6f}")
        print(f"  Median: {np.median(csif_range_distribution):.6f}")
        print(
            f"  Samples with range > {min_diff}: {np.sum(np.array(csif_range_distribution) >= min_diff)}"
        )

    # Analyze spatial distribution
    if sos_by_lat:
        print("\nSOS spatial distribution:")
        print(f"  Latitude range: {np.min(sos_by_lat):.2f} to {np.max(sos_by_lat):.2f}")
        lat_bins = pd.cut(sos_by_lat, bins=5)
        print("  Distribution by latitude band:")
        for interval, count in lat_bins.value_counts().sort_index().items():
            print(f"    {interval}: {count} samples")

    if eos_by_lat:
        print("\nEOS spatial distribution:")
        print(f"  Latitude range: {np.min(eos_by_lat):.2f} to {np.max(eos_by_lat):.2f}")
        lat_bins = pd.cut(eos_by_lat, bins=5)
        print("  Distribution by latitude band:")
        for interval, count in lat_bins.value_counts().sort_index().items():
            print(f"    {interval}: {count} samples")

    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: CSIF range distribution
    if csif_range_distribution:
        axes[0, 0].hist(csif_range_distribution, bins=50, edgecolor="black")
        axes[0, 0].axvline(
            min_diff, color="red", linestyle="--", label=f"Threshold ({min_diff})"
        )
        axes[0, 0].set_xlabel("CSIF Range")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("CSIF Range Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    # Plot 2: Slope distribution
    if slope_distribution:
        axes[0, 1].hist(slope_distribution, bins=50, edgecolor="black")
        axes[0, 1].axvline(
            min_slope,
            color="green",
            linestyle="--",
            label=f"SOS Threshold ({min_slope})",
        )
        axes[0, 1].axvline(
            -min_slope - 0.0005, color="orange", linestyle="--", label="EOS Threshold"
        )
        axes[0, 1].set_xlabel("Slope")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Slope Distribution (CSIF change rate)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    # Plot 3: SOS latitude distribution
    if sos_by_lat:
        axes[1, 0].hist(
            sos_by_lat, bins=30, edgecolor="black", color="green", alpha=0.7
        )
        axes[1, 0].set_xlabel("Latitude")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title(f"SOS Samples by Latitude (n={len(sos_by_lat)})")
        axes[1, 0].grid(alpha=0.3)

    # Plot 4: EOS latitude distribution
    if eos_by_lat:
        axes[1, 1].hist(
            eos_by_lat, bins=30, edgecolor="black", color="orange", alpha=0.7
        )
        axes[1, 1].set_xlabel("Latitude")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title(f"EOS Samples by Latitude (n={len(eos_by_lat)})")
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()

    # Save diagnostic plot
    output_path = "phenology_diagnostic.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nDiagnostic plot saved to: {output_path}")

    return SOS_indices, EOS_indices


def main():
    parser = argparse.ArgumentParser(description="Diagnose BET driver data issues")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to data pickle file"
    )
    parser.add_argument(
        "--pred-path", type=str, required=True, help="Path to predictions pickle file"
    )
    parser.add_argument(
        "--coord-path", type=str, required=True, help="Path to coordinates parquet file"
    )
    parser.add_argument(
        "--forecast-window", type=int, default=30, help="Forecast window length"
    )
    parser.add_argument(
        "--min-diff", type=float, default=0.05, help="Minimum CSIF difference threshold"
    )
    parser.add_argument(
        "--min-slope", type=float, default=0.001, help="Minimum slope threshold"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BET DRIVER DATA DIAGNOSTIC TOOL")
    print("=" * 60)

    # Run analyses
    data = analyze_data_file(args.data_path)
    preds = analyze_predictions(args.pred_path)
    coords = analyze_coordinates(args.coord_path)

    SOS_indices, EOS_indices = analyze_phenology_detection(
        data,
        preds,
        coords,
        forecast_window=args.forecast_window,
        min_diff=args.min_diff,
        min_slope=args.min_slope,
    )

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    total_detected = len(SOS_indices) + len(EOS_indices)

    if total_detected < 100:
        print("\n⚠️  LOW DETECTION RATE")
        print("\nPossible issues:")
        print("  1. Thresholds may be too strict for BET")
        print(f"     Current: min_diff={args.min_diff}, min_slope={args.min_slope}")
        print("     Try: --min-diff 0.03 --min-slope 0.0005")
        print("\n  2. BET may have weak seasonal signal")
        print("     Consider using a longer time window or different PFT")
        print("\n  3. Data quality issues")
        print("     Check if CSIF values are properly scaled")

    elif total_detected < 500:
        print("\n⚠️  MODERATE DETECTION RATE")
        print("\nThis may lead to sparse driver maps.")
        print("Consider:")
        print("  1. Using interpolation when generating maps (--interpolate flag)")
        print("  2. Slightly relaxing thresholds")
        print("  3. Increasing spatial imputation radius")

    else:
        print("\n✓ GOOD DETECTION RATE")
        print("\nDetected sufficient samples for driver maps.")
        print("If you still see NaNs:")
        print("  1. Enable interpolation (--interpolate flag)")
        print("  2. Check coordinate file coverage")
        print("  3. Increase spatial imputation radius")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
