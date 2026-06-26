#!/usr/bin/env python3
"""
Generate Attention Window Maps for Phenological Analysis

Creates global maps showing the start index of the 30-day forecast window with
maximum total attention, per plant functional type (PFT) and phenological phase
(SOS/EOS).

Unlike driver maps (RGB-encoded), these are scalar colormaps: each pixel's color
represents the window_start index (0 = early in the historical record,
~335 = just before the forecast horizon).

Input CSVs are produced by:
    Data/data_wrangling/get_attention_window_data.py

Schema expected:  location | latitude | longitude | window_start
"""

import logging
import os
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WindowMapGenerator:
    """Generate global maps of max-attention window start indices."""

    # Same lat filters as DriverMapGenerator
    LAT_FILTERS: Dict = {
        "BET": 30,
        "NET": 75,
        "NDT": 75,
        "SHR": 75,
        "GRA": 75,
        "BDT": 90,
        "BDT_5020": (20, 50),
        "BDT_2020": (-20, 20),
        "BDT_2060": (-60, -20),
        "BDT_5090": (50, 90),
    }

    # Global grid parameters (0.25° resolution)
    GRID_LATS = np.round(np.linspace(90.0, -89.75, 720), 6)
    GRID_LONS = np.round(np.linspace(-180.0, 179.75, 1440), 6)

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cmap: str = "plasma",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """
        Args:
            data_dir:   Directory containing window_start CSV files.
            output_dir: Directory to write PNG outputs.
            cmap:       Matplotlib colormap name for the scalar field.
            vmin/vmax:  Color scale limits. If None, determined from data.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax

    # ------------------------------------------------------------------
    # Data loading & processing
    # ------------------------------------------------------------------

    def load_csv_data(self, file_path: str, pft: str) -> pd.DataFrame:
        """Load and latitude-filter a window_start CSV for a given PFT."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {Path(file_path).name}")

            if pft in self.LAT_FILTERS:
                lat_filter = self.LAT_FILTERS[pft]
                if isinstance(lat_filter, tuple):
                    lo, hi = lat_filter
                    df = df[(df["latitude"] >= lo) & (df["latitude"] < hi)]
                else:
                    df = df[df["latitude"] < lat_filter]
                logger.info(f"  After lat filter: {len(df)} records")

            df["legacy_depth"] = 335 - df["window_start"]
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def build_grid(self, df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate PFT DataFrames, deduplicate, and merge onto the full global grid.

        Deduplication strategy (mirrors DriverMapGenerator):
          1. Round coordinates to 6 decimal places.
          2. Drop rows where window_start is NaN.
          3. Drop exact duplicate rows.
          4. Median-aggregate remaining coordinate duplicates.
          5. Left-join onto full 720×1440 grid (missing → NaN).

        Returns:
            DataFrame with columns [latitude, longitude, window_start],
            exactly 720×1440 = 1,036,800 rows.
        """
        if not df_list:
            logger.warning("No DataFrames to process.")
            return pd.DataFrame()

        df = pd.concat(df_list, ignore_index=True)
        df["latitude"] = df["latitude"].round(6)
        df["longitude"] = df["longitude"].round(6)
        logger.info(f"  Concatenated: {len(df)} rows")

        df = df.dropna(subset=["legacy_depth"])
        logger.info(f"  After dropping NaN legacy_depth: {len(df)} rows")

        df = df.drop_duplicates(
            subset=["latitude", "longitude", "legacy_depth"], keep="first"
        )
        logger.info(f"  After dropping exact duplicates: {len(df)} rows")

        df = df.groupby(["latitude", "longitude"], as_index=False)[
            "legacy_depth"
        ].median()
        logger.info(f"  After median aggregation: {len(df)} unique coordinates")

        # Build full grid and merge
        full_grid = pd.DataFrame(
            list(product(self.GRID_LATS, self.GRID_LONS)),
            columns=["latitude", "longitude"],
        )
        full = pd.merge(full_grid, df, on=["latitude", "longitude"], how="left")
        logger.info(
            f"  Full grid: {len(full)} cells ({full['legacy_depth'].notna().sum()} with data)"
        )
        return full

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_map(
        self,
        grid_df: pd.DataFrame,
        title: str,
        projection: str = "Robinson",
        crop_lat: float = -60,
        dpi: int = 300,
        figsize: Tuple[int, int] = (16, 8),
        save: bool = True,
        show: bool = False,
        colorbar_label: str = "Legacy depth (days before forecast horizon)",
    ) -> None:
        """
        Render a scalar window_start field on a global cartographic map.

        Args:
            grid_df:        Full 720×1440 grid DataFrame from build_grid().
            title:          Plot title and output filename stem.
            projection:     'Robinson' or 'PlateCarree'.
            crop_lat:       Hide latitudes below this value (default −60°).
            dpi:            Output resolution.
            figsize:        Figure size in inches.
            save:           Write PNG to output_dir.
            show:           Display interactively.
            colorbar_label: Label on the colorbar.
        """
        # Reshape scalar field to (720, 1440)
        data_2d = np.reshape(grid_df["legacy_depth"].values, (720, 1440))

        # Determine color scale
        valid = data_2d[~np.isnan(data_2d)]
        vmin = (
            self.vmin if self.vmin is not None else (valid.min() if len(valid) else 0)
        )
        vmax = (
            self.vmax if self.vmax is not None else (valid.max() if len(valid) else 335)
        )

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(self.cmap)

        # Map scalar → RGBA, NaN → white
        rgba = cmap_obj(norm(data_2d))  # (720, 1440, 4)
        nan_mask = np.isnan(data_2d)
        rgba[nan_mask] = [1.0, 1.0, 1.0, 1.0]  # white for missing

        # Projection
        proj = (
            ccrs.Robinson() if projection.lower() == "robinson" else ccrs.PlateCarree()
        )

        fig = plt.figure(figsize=figsize, facecolor="white")
        ax = plt.axes(projection=proj)
        ax.set_facecolor("white")
        ax.set_global()

        # Cyclic wrap for antimeridian
        lons_1d = np.linspace(-180.0, 180.0, 1440, endpoint=False)
        rgba_cyc, lon_cyc = add_cyclic_point(rgba, coord=lons_1d, axis=1)

        # Alpha mask to hide Antarctica
        H, Wc = rgba_cyc.shape[:2]
        lats_col = np.linspace(90.0, -90.0, H)[:, None]
        alpha_mask = np.repeat((lats_col >= crop_lat).astype(float), Wc, axis=1)
        rgba_cyc[..., 3] *= alpha_mask

        ax.imshow(
            rgba_cyc,
            extent=[lon_cyc.min(), lon_cyc.max(), -90.0, 90.0],
            origin="upper",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
        )

        if crop_lat is not None and projection.lower() != "robinson":
            ax.set_extent(
                [lon_cyc.min(), lon_cyc.max(), crop_lat, 90.0],
                crs=ccrs.PlateCarree(),
            )

        ax.coastlines(resolution="110m", color="black", linewidth=0.5)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.03, shrink=0.6
        )
        cbar.set_label(colorbar_label, fontsize=11)

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        if save and title:
            out_path = self.output_dir / f"{title}.png"
            plt.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Saved: {out_path}")

        plt.show() if show else plt.close(fig)

    # ------------------------------------------------------------------
    # High-level generation methods
    # ------------------------------------------------------------------

    def generate_pft_map(
        self,
        file_paths: List[str],
        pft_names: List[str],
        output_name: str,
        **plot_kwargs,
    ) -> None:
        """Generate a window-start map for one or more PFT CSV files."""
        dataframes = []
        for fp, pft in zip(file_paths, pft_names):
            if not os.path.exists(fp):
                logger.warning(f"File not found: {fp}")
                continue
            df = self.load_csv_data(fp, pft)
            if not df.empty:
                dataframes.append(df)

        if not dataframes:
            logger.error(f"No valid data for {output_name}")
            return

        grid_df = self.build_grid(dataframes)
        self.plot_map(grid_df, output_name, **plot_kwargs)
        logger.info(f"Generated: {output_name}")

    def generate_multi_panel_map(
        self,
        file_mapping: Dict[str, Dict[str, str]],
        phase: str,
        projection: str = "Robinson",
        crop_lat: float = -60,
        dpi: int = 300,
        save: bool = True,
        show: bool = False,
    ) -> None:
        """
        Generate a 2×3 multi-panel figure with one subplot per PFT.

        Args:
            file_mapping: {pft: {phase: filepath}} dict from create_file_mapping().
            phase:        'SOS' or 'EOS'.
        """
        pfts = ["BET", "NET", "NDT", "BDT", "SHR", "GRA"]
        proj = (
            ccrs.Robinson() if projection.lower() == "robinson" else ccrs.PlateCarree()
        )

        fig = plt.figure(figsize=(24, 16), facecolor="white")

        # Collect all valid data first to compute a shared color scale
        all_grids: Dict[str, pd.DataFrame] = {}
        for pft in pfts:
            if pft not in file_mapping or phase not in file_mapping[pft]:
                continue
            df = self.load_csv_data(file_mapping[pft][phase], pft)
            if not df.empty:
                all_grids[pft] = self.build_grid([df])

        if not all_grids:
            logger.error(f"No data found for multi-panel {phase} map.")
            return

        # Shared color scale across all panels
        vmin = self.vmin
        vmax = self.vmax
        if vmin is None or vmax is None:
            all_vals = np.concatenate(
                [g["legacy_depth"].dropna().values for g in all_grids.values()]
            )
            vmin = vmin if vmin is not None else all_vals.min()
            vmax = vmax if vmax is not None else all_vals.max()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(self.cmap)

        axes_list = []
        for idx, pft in enumerate(pfts, 1):
            if pft not in all_grids:
                logger.warning(f"Skipping {pft} (no data)")
                continue

            grid_df = all_grids[pft]
            data_2d = np.reshape(grid_df["legacy_depth"].values, (720, 1440))

            rgba = cmap_obj(norm(data_2d))
            rgba[np.isnan(data_2d)] = [1.0, 1.0, 1.0, 1.0]

            ax = fig.add_subplot(2, 3, idx, projection=proj)
            ax.set_facecolor("white")
            ax.set_global()

            lons_1d = np.linspace(-180.0, 180.0, 1440, endpoint=False)
            rgba_cyc, lon_cyc = add_cyclic_point(rgba, coord=lons_1d, axis=1)

            H, Wc = rgba_cyc.shape[:2]
            lats_col = np.linspace(90.0, -90.0, H)[:, None]
            alpha_mask = np.repeat((lats_col >= crop_lat).astype(float), Wc, axis=1)
            rgba_cyc[..., 3] *= alpha_mask

            ax.imshow(
                rgba_cyc,
                extent=[lon_cyc.min(), lon_cyc.max(), -90.0, 90.0],
                origin="upper",
                transform=ccrs.PlateCarree(),
                interpolation="nearest",
            )
            if projection.lower() != "robinson":
                ax.set_extent(
                    [lon_cyc.min(), lon_cyc.max(), crop_lat, 90.0],
                    crs=ccrs.PlateCarree(),
                )
            ax.coastlines(resolution="110m", color="black", linewidth=0.5)
            ax.set_title(f"{pft} — {phase}", fontsize=13, fontweight="bold")
            axes_list.append(ax)

        # Shared colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes_list,
            orientation="horizontal",
            pad=0.04,
            fraction=0.02,
            shrink=0.5,
        )
        cbar.set_label("Legacy depth (days before forecast horizon)", fontsize=12)

        fig.suptitle(
            f"Climate Legacy Depth — All PFTs ({phase})\n"
            f"Days before forecast horizon that capture peak attention\n"
            f"(0 = current season only;  335 = prior-year climate drives phenology)",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            out_path = self.output_dir / f"All_PFTs_MultiPanel_LegacyDepth_{phase}.png"
            plt.savefig(
                out_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            logger.info(f"Saved: {out_path}")

        plt.show() if show else plt.close(fig)


# ------------------------------------------------------------------
# File discovery (mirrors create_file_mapping in generate_driver_maps.py)
# ------------------------------------------------------------------


def create_file_mapping(data_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Auto-discover window_start CSV files in data_dir.

    Expects filenames like:  {PFT}_SOS.csv  or  {PFT}_window_data_SOS.csv
    Returns: {pft: {phase: filepath}}
    """
    data_path = Path(data_dir)
    patterns = {
        "BET": {"SOS": "BET_*SOS*.csv", "EOS": "BET_*EOS*.csv"},
        "NET": {"SOS": "NET_*SOS*.csv", "EOS": "NET_*EOS*.csv"},
        "NDT": {"SOS": "NDT_*SOS*.csv", "EOS": "NDT_*EOS*.csv"},
        "SHR": {"SOS": "SHR_*SOS*.csv", "EOS": "SHR_*EOS*.csv"},
        "GRA": {"SOS": "GRA_*SOS*.csv", "EOS": "GRA_*EOS*.csv"},
        "BDT": {"SOS": "BDT_*SOS*.csv", "EOS": "BDT_*EOS*.csv"},
        "BDT_5020": {"SOS": "BDT_50_20_*SOS*.csv", "EOS": "BDT_50_20_*EOS*.csv"},
        "BDT_2020": {"SOS": "BDT_-20_20_*SOS*.csv", "EOS": "BDT_-20_20_*EOS*.csv"},
        "BDT_2060": {"SOS": "BDT_-20_-60_*SOS*.csv", "EOS": "BDT_-20_-60_*EOS*.csv"},
        "BDT_5090": {"SOS": "BDT_50_90_*SOS*.csv", "EOS": "BDT_50_90_*EOS*.csv"},
    }

    mapping: Dict[str, Dict[str, str]] = {}
    for pft, phases in patterns.items():
        mapping[pft] = {}
        for phase, pattern in phases.items():
            matches = sorted(data_path.glob(pattern))
            if matches:
                mapping[pft][phase] = str(matches[-1])

    return mapping


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate global maps of max-attention window start indices"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing window_start CSV files"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save output PNGs"
    )
    parser.add_argument(
        "--projection", choices=["PlateCarree", "Robinson"], default="Robinson"
    )
    parser.add_argument(
        "--cmap", default="plasma", help="Matplotlib colormap name (default: plasma)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Color scale minimum (default: data min)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color scale maximum (default: data max)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-plots", action="store_true")

    args = parser.parse_args()

    gen = WindowMapGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    file_mapping = create_file_mapping(args.data_dir)

    plot_kwargs = dict(
        projection=args.projection,
        crop_lat=-60,
        dpi=args.dpi,
        show=args.show_plots,
    )

    # Individual PFT maps
    for pft in ["BET", "NET", "NDT", "SHR", "GRA"]:
        for phase in ["SOS", "EOS"]:
            if pft in file_mapping and phase in file_mapping[pft]:
                gen.generate_pft_map(
                    [file_mapping[pft][phase]],
                    [pft],
                    f"{pft}_LegacyDepth_{phase}",
                    **plot_kwargs,
                )

    # Combined BDT map (four latitudinal bands merged)
    for phase in ["SOS", "EOS"]:
        bdt_files, bdt_names = [], []
        for bdt_type in ["BDT_5020", "BDT_2020", "BDT_2060", "BDT_5090"]:
            if bdt_type in file_mapping and phase in file_mapping[bdt_type]:
                bdt_files.append(file_mapping[bdt_type][phase])
                bdt_names.append(bdt_type)
        if bdt_files:
            gen.generate_pft_map(
                bdt_files,
                bdt_names,
                f"BDT_LegacyDepth_{phase}",
                **plot_kwargs,
            )
        elif "BDT" in file_mapping and phase in file_mapping["BDT"]:
            gen.generate_pft_map(
                [file_mapping["BDT"][phase]],
                ["BDT"],
                f"BDT_LegacyDepth_{phase}",
                **plot_kwargs,
            )

    # All-PFTs combined map
    for phase in ["SOS", "EOS"]:
        all_files, all_names = [], []
        for pft, phases in file_mapping.items():
            if phase in phases:
                all_files.append(phases[phase])
                all_names.append(pft)
        if all_files:
            gen.generate_pft_map(
                all_files,
                all_names,
                f"All_PFTs_LegacyDepth_{phase}",
                **plot_kwargs,
            )

    # Multi-panel map (one subplot per PFT, shared colorbar)
    for phase in ["SOS", "EOS"]:
        gen.generate_multi_panel_map(
            file_mapping,
            phase,
            projection=args.projection,
            crop_lat=-60,
            dpi=args.dpi,
            show=args.show_plots,
        )

    logger.info("Window map generation complete.")


if __name__ == "__main__":
    main()
