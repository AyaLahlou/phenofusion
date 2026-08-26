#!/usr/bin/env python3
"""
Fill equator-band gaps in per-location attention CSVs.

Several PFT CSVs (both the max-attention-window `window_start` column and
the attention-weighted `mean_attention_day` column) show a band of
near-total data dropout at latitudes close to 0 -- an artifact of the
equator-crossing latitude-band boundary in the upstream extraction -- which
shows up as a blank/noisy stripe in the rendered maps. Coverage elsewhere
is dense and unaffected.

For each row whose latitude falls in the anomalous band and whose value is
NaN, this fills it with the median of its k nearest valid neighbors
(great-circle distance, via a unit-sphere KDTree so the antimeridian and
poles are handled correctly). The anomalous band is auto-detected per file:
0.25-degree latitude bins within `--equator-window` degrees of the equator
whose missing rate exceeds `--threshold` are treated as part of the gap.

Generalizes `fill_attention_window_gaps.py` (which hardcodes the
"window_start" column and a fixed BET-only latitude list) to any value
column and to auto-detected bands, so the same fix applies to
`data/attention_mean/*.csv` as well as `data/attention_window/*.csv`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def detect_equator_band(
    df: pd.DataFrame,
    value_col: str,
    equator_window: float,
    threshold: float,
    bin_size: float = 0.25,
) -> list:
    """Return the sorted list of latitude bin centers near the equator whose
    missing rate for `value_col` exceeds `threshold` (fraction, e.g. 0.4)."""
    near = df[df["latitude"].abs() <= equator_window].copy()
    if near.empty:
        return []
    near["latbin"] = (near["latitude"] / bin_size).round() * bin_size
    stats = near.groupby("latbin")[value_col].agg(
        n="size", missing=lambda s: s.isna().sum()
    )
    stats["rate"] = stats["missing"] / stats["n"]
    return sorted(stats[stats["rate"] > threshold].index.tolist())


def fill_gaps(
    df: pd.DataFrame, value_col: str, target_lats: list, k: int, tol: float = 1e-6
) -> pd.DataFrame:
    df = df.copy()
    if not target_lats:
        return df
    is_target = df["latitude"].apply(
        lambda lat: any(abs(lat - t) < tol for t in target_lats)
    )
    missing = is_target & df[value_col].isna()

    valid = df[value_col].notna()
    donor_pool = valid & ~missing  # all valid points not being filled

    if missing.sum() == 0:
        return df

    tree = cKDTree(
        latlon_to_xyz(
            df.loc[donor_pool, "latitude"].values,
            df.loc[donor_pool, "longitude"].values,
        )
    )
    donor_values = df.loc[donor_pool, value_col].values

    query_xyz = latlon_to_xyz(
        df.loc[missing, "latitude"].values, df.loc[missing, "longitude"].values
    )
    _, idx = tree.query(query_xyz, k=k)
    idx = np.atleast_2d(idx)

    fill_values = np.median(donor_values[idx], axis=1)
    df.loc[missing, value_col] = fill_values
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect and fill equator-band gaps in attention CSVs"
    )
    parser.add_argument("csv_paths", nargs="+", help="CSV files to fill in place")
    parser.add_argument(
        "--value-col",
        default="mean_attention_day",
        help="Column to fill (default: mean_attention_day)",
    )
    parser.add_argument(
        "--equator-window",
        type=float,
        default=3.0,
        help="Search for the anomalous band within +/- this many degrees of the equator (default: 3.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Latitude bins with missing rate above this fraction are treated as anomalous (default: 0.4)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of nearest neighbors for the median fill",
    )
    args = parser.parse_args()

    for path_str in args.csv_paths:
        path = Path(path_str)
        df = pd.read_csv(path)
        if args.value_col not in df.columns:
            print(f"{path}: no '{args.value_col}' column, skipping")
            continue

        target_lats = detect_equator_band(
            df, args.value_col, args.equator_window, args.threshold
        )
        n_before = df[args.value_col].isna().sum()
        filled = fill_gaps(df, args.value_col, target_lats, args.k)
        n_after = filled[args.value_col].isna().sum()
        filled.to_csv(path, index=False)
        print(
            f"{path}: band={target_lats} filled {n_before - n_after} rows "
            f"({n_before} -> {n_after} still NaN)"
        )


if __name__ == "__main__":
    main()
