#!/usr/bin/env python3
"""
Sensitivity-shift analysis and plotting.

Builds the maps that compare TFT-derived climate driver sensitivities between
two epochs (e.g. 1982-1991 vs 2012-2021). All figures are powered by per-
(location, phase, year) records produced by
`Data/data_wrangling/extract_epoch_drivers.py`.

Two top-level classes:

    SensitivityShiftAnalyzer
        Loads yearly CSVs, applies PFT latitude filters, applies the canonical
        dedup pipeline, computes per-pixel epoch medians, runs chunked
        bootstrap for confidence intervals, derives signed Δ values and
        dominance-flip categories.

    SensitivityShiftPlotter
        Renders Tier 1/2/3 figures from the analyzer's outputs.

Notes on choices that differ from the snapshot pipeline
-------------------------------------------------------
* Solar composite uses ``hist_rad + hist_photo`` (the photoperiod term that
  ``generate_driver_maps.py`` line 155 omits). Toggle with ``legacy_sol=True``
  to fall back to ``2 * hist_rad`` for back-compat with already-published
  figures.
* Epochs are pixel-matched via an *inner* join, so a pixel only contributes
  to Δ maps if it has data in both epochs. The headline RGB pipeline left-
  joins onto the full grid; that's fine for a single snapshot but propagates
  NaNs into Δ.
* Bootstrap is computed locally in chunks (default B=200) over pixels that
  have ≥1 sample in each epoch.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

DRIVER_NAMES: Tuple[str, ...] = ("tmin", "tmax", "rad", "precip", "photo", "sm")
DRIVER_COLS: Tuple[str, ...] = tuple(f"hist_{d}" for d in DRIVER_NAMES)
GROUPS: Tuple[str, ...] = ("temp", "sol", "p")
GROUP_LABELS: Dict[str, str] = {
    "temp": "Temperature (Tmin+Tmax)",
    "sol": "Solar (Rad+Photo)",
    "p": "Water (Precip+SoilMoist)",
}
GROUP_COLORS: Dict[str, str] = {"temp": "#d62728", "sol": "#2ca02c", "p": "#1f77b4"}

# Same lat filters used by the snapshot pipeline so figures align.
LAT_FILTERS: Dict[str, object] = {
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

GRID_LATS: np.ndarray = np.round(np.linspace(90.0, -89.75, 720), 6)
GRID_LONS: np.ndarray = np.round(np.linspace(-180.0, 179.75, 1440), 6)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Epoch:
    label: str
    year_start: int
    year_end: int  # inclusive

    def contains(self, year: int) -> bool:
        return self.year_start <= year <= self.year_end


@dataclass
class ShiftConfig:
    """Top-level configuration for the sensitivity-shift analysis."""

    early: Epoch = Epoch("1982-1991", 1982, 1991)
    late: Epoch = Epoch("2012-2021", 2012, 2021)
    legacy_sol: bool = False  # if True, sol = 2*rad (snapshot bug)
    bootstrap_B: int = 200
    bootstrap_chunk: int = 4000  # pixels per chunk
    rng_seed: int = 20260603
    crop_lat: float = -60.0
    projection: str = "Robinson"
    dpi: int = 300

    # Rolling-decade trajectory (Tier 3)
    rolling_width: int = 10
    rolling_step: int = 5
    rolling_years: Tuple[int, int] = (1982, 2021)

    pfts: Tuple[str, ...] = ("BET", "NET", "NDT", "BDT", "SHR", "GRA")

    # Diverging color scale clip for per-driver Δ maps.
    diverging_vmax: float = 0.15  # absolute cap on |Δ|

    # Spatial aggregation before rendering mosaic maps (degrees). 1.0 means each
    # 1°×1° cell shows the median of all 0.25° data points within it, filling
    # gaps caused by sparse PFT biome coverage. Set to 0.25 to skip aggregation.
    coarsen_resolution: float = 1.0

    # IDW fill radius (degrees, Euclidean in lat/lon space). After coarsening,
    # empty cells within this distance of any real data cell are filled by
    # inverse-distance-weighted interpolation. Set to 0 to disable.
    # 1.1° = fill only the immediate N/S/E/W neighbours of real cells (minimum
    # useful value for a 1° coarsened grid; diagonal gaps need √2°≈1.41°).
    idw_radius: float = 1.1

    extra: Dict[str, object] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _composite_groups(df: pd.DataFrame, legacy_sol: bool) -> pd.DataFrame:
    """Return df with added temp/sol/p columns, row-normalized to sum to 1."""
    out = df.copy()
    out["temp"] = out["hist_tmin"] + out["hist_tmax"]
    out["sol"] = (
        (2.0 * out["hist_rad"]) if legacy_sol else (out["hist_rad"] + out["hist_photo"])
    )
    out["p"] = out["hist_precip"] + out["hist_sm"]
    total = out[["temp", "sol", "p"]].sum(axis=1)
    safe = total.replace(0.0, np.nan)
    out[["temp", "sol", "p"]] = out[["temp", "sol", "p"]].div(safe, axis=0)
    return out


def _lat_filter(df: pd.DataFrame, pft: str) -> pd.DataFrame:
    f = LAT_FILTERS.get(pft)
    if f is None:
        return df
    if isinstance(f, tuple):
        lo, hi = f
        return df[(df["latitude"] >= lo) & (df["latitude"] < hi)]
    return df[df["latitude"] < f]


def _full_grid_index() -> pd.DataFrame:
    """A 720x1440 (lat, lon) DataFrame for left-joins."""
    return pd.DataFrame(
        list(product(GRID_LATS, GRID_LONS)), columns=["latitude", "longitude"]
    )


def _round_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["latitude"] = df["latitude"].round(6)
    df["longitude"] = df["longitude"].round(6)
    return df


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------


class SensitivityShiftAnalyzer:
    """Load yearly CSVs and derive per-pixel epoch medians, Δ values, CIs."""

    def __init__(self, data_dir: Path, config: Optional[ShiftConfig] = None):
        self.data_dir = Path(data_dir)
        self.config = config or ShiftConfig()
        self._raw_cache: Dict[str, pd.DataFrame] = {}
        self._grouped_cache: Dict[str, pd.DataFrame] = {}
        self._rng = np.random.default_rng(self.config.rng_seed)

    # ---- I/O ----------------------------------------------------------

    def _load_pft_raw(self, pft: str) -> pd.DataFrame:
        if pft in self._raw_cache:
            return self._raw_cache[pft]

        candidates = sorted(self.data_dir.glob(f"{pft}_yearly_drivers*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"No yearly driver CSV for {pft} in {self.data_dir}. "
                "Did you run Data/data_wrangling/extract_epoch_drivers.py on HPC and rsync the result?"
            )
        path = candidates[-1]
        logger.info(f"[{pft}] reading {path.name}")
        df = pd.read_csv(path)

        required = {"location", "latitude", "longitude", "phase", "year", *DRIVER_COLS}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = _round_coords(df)
        df = _lat_filter(df, pft)
        df = df.dropna(subset=list(DRIVER_COLS), how="all")
        self._raw_cache[pft] = df
        return df

    def _load_pft_grouped(self, pft: str) -> pd.DataFrame:
        """Raw with per-row temp/sol/p composites added (row-normalized)."""
        if pft in self._grouped_cache:
            return self._grouped_cache[pft]
        df = _composite_groups(self._load_pft_raw(pft), self.config.legacy_sol)
        self._grouped_cache[pft] = df
        return df

    # ---- Epoch tables -------------------------------------------------

    def epoch_pixel_table(
        self,
        pft: str,
        phase: str,
        epoch: Epoch,
        full_grid: bool = False,
    ) -> pd.DataFrame:
        """Median composite weights per pixel for the requested epoch + phase.

        Returns columns ``[latitude, longitude, temp, sol, p, n_samples]``.
        With ``full_grid=False`` (default), only pixels that had ≥1 sample are
        returned. With ``full_grid=True``, the table is left-joined onto the
        720x1440 global grid (missing pixels = NaN).
        """
        df = self._load_pft_grouped(pft)
        mask = (df["phase"] == phase) & (
            df["year"].between(epoch.year_start, epoch.year_end)
        )
        sub = df.loc[mask, ["latitude", "longitude", "temp", "sol", "p"]]
        if sub.empty:
            logger.warning(f"[{pft}/{phase}/{epoch.label}] no rows in epoch.")
            return sub.assign(n_samples=0)

        agg = sub.groupby(["latitude", "longitude"], as_index=False).agg(
            temp=("temp", "median"),
            sol=("sol", "median"),
            p=("p", "median"),
            n_samples=("temp", "size"),
        )
        if not full_grid:
            return agg
        return _full_grid_index().merge(agg, on=["latitude", "longitude"], how="left")

    # ---- Δ + bootstrap CI --------------------------------------------

    def epoch_delta(
        self,
        pft: str,
        phase: str,
        with_ci: bool = True,
    ) -> pd.DataFrame:
        """Δ = late − early per pixel, inner-joined.

        Columns: ``latitude, longitude, n_early, n_late,
                  temp_early, sol_early, p_early,
                  temp_late, sol_late, p_late,
                  dtemp, dsol, dp,
                  dtemp_lo, dtemp_hi, dsol_lo, dsol_hi, dp_lo, dp_hi``  (CIs if requested)
        """
        early = self.epoch_pixel_table(pft, phase, self.config.early).rename(
            columns={c: f"{c}_early" for c in ("temp", "sol", "p")}
            | {"n_samples": "n_early"}
        )
        late = self.epoch_pixel_table(pft, phase, self.config.late).rename(
            columns={c: f"{c}_late" for c in ("temp", "sol", "p")}
            | {"n_samples": "n_late"}
        )
        if early.empty or late.empty:
            logger.warning(f"[{pft}/{phase}] empty epoch table; returning empty Δ.")
            return pd.DataFrame()

        merged = early.merge(late, on=["latitude", "longitude"], how="inner")
        for g in GROUPS:
            merged[f"d{g}"] = merged[f"{g}_late"] - merged[f"{g}_early"]

        if not with_ci:
            return merged

        # Bootstrap CIs on Δ medians
        df_g = self._load_pft_grouped(pft)
        merged = self._attach_bootstrap_ci(df_g, phase, merged)
        return merged

    # ---- Bootstrap ----------------------------------------------------

    def _attach_bootstrap_ci(
        self,
        df_grouped: pd.DataFrame,
        phase: str,
        delta_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Chunked paired-bootstrap of Δ-median 95% CIs."""

        cfg = self.config
        B = cfg.bootstrap_B
        early_yrs = (cfg.early.year_start, cfg.early.year_end)
        late_yrs = (cfg.late.year_start, cfg.late.year_end)

        ph = df_grouped[df_grouped["phase"] == phase]
        early = ph[ph["year"].between(*early_yrs)]
        late = ph[ph["year"].between(*late_yrs)]

        # Build per-(lat,lon) lists of sample weights (3-tuples) as numpy arrays.
        def _group_array(df: pd.DataFrame) -> Dict[Tuple[float, float], np.ndarray]:
            arr = df[["latitude", "longitude", "temp", "sol", "p"]].to_numpy()
            keys = list(zip(arr[:, 0], arr[:, 1]))
            order = np.lexsort((arr[:, 1], arr[:, 0]))
            arr_sorted = arr[order]
            keys_sorted = [keys[i] for i in order]

            out: Dict[Tuple[float, float], np.ndarray] = {}
            i = 0
            n = len(arr_sorted)
            while i < n:
                k = keys_sorted[i]
                j = i + 1
                while j < n and keys_sorted[j] == k:
                    j += 1
                out[k] = arr_sorted[i:j, 2:5].astype(np.float32)
                i = j
            return out

        logger.info(
            f"  bootstrap: indexing early ({len(early)} rows) and late ({len(late)} rows)"
        )
        early_map = _group_array(early)
        late_map = _group_array(late)

        # Iterate over inner-joined pixels in chunks.
        keys = [
            (lat, lon)
            for lat, lon in zip(
                delta_table["latitude"].to_numpy(), delta_table["longitude"].to_numpy()
            )
        ]

        ci_cols = {
            f"d{g}_{end}": np.full(len(keys), np.nan, dtype=np.float32)
            for g in GROUPS
            for end in ("lo", "hi")
        }

        chunk_size = cfg.bootstrap_chunk
        rng = self._rng
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start : start + chunk_size]
            for offset, key in enumerate(chunk):
                e = early_map.get(key)
                late = late_map.get(key)
                if e is None or late is None or len(e) == 0 or len(late) == 0:
                    continue

                # Independent bootstrap per epoch.
                e_idx = rng.integers(0, len(e), size=(B, len(e)))
                l_idx = rng.integers(0, len(late), size=(B, len(late)))
                e_samp = np.median(e[e_idx], axis=1)  # (B, 3)
                l_samp = np.median(late[l_idx], axis=1)  # (B, 3)
                d_samp = l_samp - e_samp  # (B, 3)
                lo = np.percentile(d_samp, 2.5, axis=0)
                hi = np.percentile(d_samp, 97.5, axis=0)
                row = start + offset
                for gi, g in enumerate(GROUPS):
                    ci_cols[f"d{g}_lo"][row] = lo[gi]
                    ci_cols[f"d{g}_hi"][row] = hi[gi]

            if (start // chunk_size) % 10 == 0:
                logger.info(
                    f"  bootstrap chunk {start // chunk_size + 1} / "
                    f"{(len(keys) + chunk_size - 1) // chunk_size}"
                )

        for col, vals in ci_cols.items():
            delta_table[col] = vals
        for g in GROUPS:
            delta_table[f"d{g}_sig"] = (delta_table[f"d{g}_lo"] > 0) | (
                delta_table[f"d{g}_hi"] < 0
            )
        return delta_table

    # ---- Dominance flip ----------------------------------------------

    @staticmethod
    def dominance_codes(table: pd.DataFrame) -> pd.DataFrame:
        """Argmax of {temp_early, sol_early, p_early} and likewise late.

        Returns columns ``early_dom`` and ``late_dom`` in {0=temp, 1=sol, 2=p}
        and ``flip_code = 3*early + late`` ∈ {0..8}.
        """
        if table.empty:
            return table
        e = table[["temp_early", "sol_early", "p_early"]].to_numpy()
        late = table[["temp_late", "sol_late", "p_late"]].to_numpy()
        early_dom = np.argmax(e, axis=1)
        late_dom = np.argmax(late, axis=1)
        out = table.copy()
        out["early_dom"] = early_dom
        out["late_dom"] = late_dom
        out["flip_code"] = 3 * early_dom + late_dom
        # Margin = top - second
        e_sorted = np.sort(e, axis=1)
        l_sorted = np.sort(late, axis=1)
        out["early_margin"] = e_sorted[:, -1] - e_sorted[:, -2]
        out["late_margin"] = l_sorted[:, -1] - l_sorted[:, -2]
        out["dmargin"] = out["late_margin"] - out["early_margin"]
        return out

    # ---- Zonal profile (Tier 2) --------------------------------------

    def zonal_profile(
        self, pft: str, phase: str, delta_table: pd.DataFrame, lat_step: float = 1.0
    ) -> pd.DataFrame:
        """Latitude-binned median Δ per driver with IQR."""
        if delta_table.empty:
            return pd.DataFrame()
        df = delta_table.copy()
        df["lat_bin"] = (df["latitude"] // lat_step) * lat_step
        rows = []
        for lat_bin, sub in df.groupby("lat_bin"):
            row = {"lat_bin": lat_bin, "n_pixels": len(sub)}
            for g in GROUPS:
                vals = sub[f"d{g}"].dropna().to_numpy()
                if len(vals) == 0:
                    row.update(
                        {
                            f"d{g}_median": np.nan,
                            f"d{g}_q25": np.nan,
                            f"d{g}_q75": np.nan,
                        }
                    )
                else:
                    row[f"d{g}_median"] = np.median(vals)
                    row[f"d{g}_q25"] = np.percentile(vals, 25)
                    row[f"d{g}_q75"] = np.percentile(vals, 75)
            rows.append(row)
        return pd.DataFrame(rows).sort_values("lat_bin").reset_index(drop=True)

    # ---- Rolling decadal trajectory (Tier 3) -------------------------

    def rolling_zonal(
        self, pft: str, phase: str, lat_step: float = 5.0
    ) -> pd.DataFrame:
        """Rolling-window zonal-mean composite weights for the trajectory plot.

        Returns long-format rows
            ``[window_center, lat_bin, group, value]``
        """
        cfg = self.config
        df = self._load_pft_grouped(pft)
        df = df[df["phase"] == phase]
        if df.empty:
            return pd.DataFrame()

        df = df.assign(lat_bin=(df["latitude"] // lat_step) * lat_step)
        starts = range(
            cfg.rolling_years[0],
            cfg.rolling_years[1] - cfg.rolling_width + 2,
            cfg.rolling_step,
        )

        out_rows = []
        for ys in starts:
            ye = ys + cfg.rolling_width - 1
            sub = df[df["year"].between(ys, ye)]
            if sub.empty:
                continue
            centre = ys + cfg.rolling_width / 2.0 - 0.5
            agg = sub.groupby("lat_bin", as_index=False)[["temp", "sol", "p"]].median()
            for _, row in agg.iterrows():
                for g in GROUPS:
                    out_rows.append(
                        {
                            "window_center": centre,
                            "window_start": ys,
                            "window_end": ye,
                            "lat_bin": row["lat_bin"],
                            "group": g,
                            "value": row[g],
                        }
                    )
        return pd.DataFrame(out_rows)

    # ---- Summary table (Tier 2) --------------------------------------

    def summary_row(
        self, pft: str, phase: str, delta_table: pd.DataFrame
    ) -> Dict[str, float]:
        """Area-weighted mean Δ + significant-fraction per driver group."""
        if delta_table.empty:
            return {
                "pft": pft,
                "phase": phase,
                **{f"mean_d{g}": np.nan for g in GROUPS},
                **{f"frac_sig_pos_d{g}": np.nan for g in GROUPS},
                **{f"frac_sig_neg_d{g}": np.nan for g in GROUPS},
                "n_pixels": 0,
            }

        weights = np.cos(np.deg2rad(delta_table["latitude"].to_numpy()))
        weights = np.clip(weights, 0, None)
        row = {"pft": pft, "phase": phase, "n_pixels": len(delta_table)}
        for g in GROUPS:
            vals = delta_table[f"d{g}"].to_numpy()
            sig = delta_table.get(f"d{g}_sig")
            wsum = np.nansum(weights * vals)
            wnorm = np.nansum(weights[~np.isnan(vals)])
            row[f"mean_d{g}"] = wsum / wnorm if wnorm > 0 else np.nan
            if sig is not None:
                sig_arr = sig.to_numpy()
                row[f"frac_sig_pos_d{g}"] = float(np.mean(sig_arr & (vals > 0)))
                row[f"frac_sig_neg_d{g}"] = float(np.mean(sig_arr & (vals < 0)))
            else:
                row[f"frac_sig_pos_d{g}"] = np.nan
                row[f"frac_sig_neg_d{g}"] = np.nan
        return row

    # ---- Combine PFTs into a global mosaic ---------------------------

    def mosaic_delta(self, phase: str, with_ci: bool = True) -> pd.DataFrame:
        """Concatenate per-PFT delta tables, then median-aggregate overlaps.

        Used by the all-PFT mosaic Tier 1 figures and the dominance-flip
        mosaic. Returns the same columns as ``epoch_delta`` with one row per
        (lat, lon) cell.
        """
        parts = []
        for pft in self.config.pfts:
            try:
                d = self.epoch_delta(pft, phase, with_ci=with_ci)
            except FileNotFoundError as e:
                logger.warning(str(e))
                continue
            if d.empty:
                continue
            d = d.assign(_pft=pft)
            parts.append(d)
        if not parts:
            return pd.DataFrame()
        cat = pd.concat(parts, ignore_index=True)
        agg_cols = [
            c
            for c in cat.columns
            if c not in ("latitude", "longitude", "_pft")
            and pd.api.types.is_numeric_dtype(cat[c])
        ]
        return cat.groupby(["latitude", "longitude"], as_index=False)[agg_cols].median()


# ----------------------------------------------------------------------
# Plotter
# ----------------------------------------------------------------------


class SensitivityShiftPlotter:
    """Render Tier 1/2/3 figures from analyzer outputs."""

    # Diverging colormap for signed Δ
    DELTA_CMAP = "RdBu_r"

    # 9-category dominance-flip palette. Indices: 3*early + late, with
    # 0=temp(R), 1=sol(G), 2=p(B).
    FLIP_PALETTE: Tuple[str, ...] = (
        "#fcd9d6",  # 0 T->T  (muted red)
        "#d96d2f",  # 1 T->S  (orange-red, temp→solar)
        "#7e3f9f",  # 2 T->P  (purple, temp→water)
        "#e5e547",  # 3 S->T  (yellow, solar→temp)
        "#1e8a2b",  # 4 S->S  (vivid dark green)
        "#004d40",  # 5 S->P  (very dark teal, solar→water)
        "#c25b8b",  # 6 P->T  (pink, water→temp)
        "#6ec6e8",  # 7 P->S  (light sky blue, water→solar)
        "#1565c0",  # 8 P->P  (vivid deep blue)
    )
    FLIP_LABELS: Tuple[str, ...] = (
        "T→T",
        "T→S",
        "T→P",
        "S→T",
        "S→S",
        "S→P",
        "P→T",
        "P→S",
        "P→P",
    )

    def __init__(self, analyzer: SensitivityShiftAnalyzer, output_dir: Path):
        self.an = analyzer
        self.cfg = analyzer.config
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ---- Shared canvas helpers ---------------------------------------

    def _proj(self):
        return (
            ccrs.Robinson()
            if self.cfg.projection.lower() == "robinson"
            else ccrs.PlateCarree()
        )

    def _setup_ax(self, ax, title: Optional[str] = None) -> None:
        ax.set_facecolor("white")
        ax.set_global()
        ax.coastlines(resolution="110m", color="black", linewidth=0.5)
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")

    def _crop_extent(self, ax, lon_min: float, lon_max: float) -> None:
        ax.set_extent(
            [lon_min, lon_max, self.cfg.crop_lat, 90.0], crs=ccrs.PlateCarree()
        )

    def _grid_2d(self, table: pd.DataFrame, col: str) -> np.ndarray:
        """Place a column from an irregular (lat,lon) table onto the 720x1440 grid."""
        full = _full_grid_index().merge(
            table[["latitude", "longitude", col]],
            on=["latitude", "longitude"],
            how="left",
        )
        return full[col].to_numpy().reshape(720, 1440)

    def _imshow_continuous(self, ax, data_2d: np.ndarray, norm, cmap: str):
        rgba = plt.get_cmap(cmap)(norm(data_2d))
        rgba[np.isnan(data_2d)] = [1.0, 1.0, 1.0, 1.0]
        lons_1d = np.linspace(-180.0, 180.0, 1440, endpoint=False)
        rgba_cyc, lon_cyc = add_cyclic_point(rgba, coord=lons_1d, axis=1)
        H, Wc = rgba_cyc.shape[:2]
        lats_col = np.linspace(90.0, -90.0, H)[:, None]
        alpha_mask = np.repeat(
            (lats_col >= self.cfg.crop_lat).astype(float), Wc, axis=1
        )
        rgba_cyc[..., 3] *= alpha_mask
        ax.imshow(
            rgba_cyc,
            extent=[lon_cyc.min(), lon_cyc.max(), -90.0, 90.0],
            origin="upper",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
        )
        self._crop_extent(ax, lon_cyc.min(), lon_cyc.max())

    def _stipple(self, ax, sig_2d: np.ndarray, density: int = 6) -> None:
        """Overlay stippling dots where sig_2d is True. Decimated to reduce clutter."""
        if sig_2d is None or not sig_2d.any():
            return
        lat_idx, lon_idx = np.where(sig_2d)
        if len(lat_idx) == 0:
            return
        keep = (lat_idx % density == 0) & (lon_idx % density == 0)
        lats = 90.0 - lat_idx[keep] * 0.25
        lons = -180.0 + lon_idx[keep] * 0.25
        ax.scatter(
            lons,
            lats,
            s=0.2,
            c="black",
            alpha=0.45,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    # ==================================================================
    # Tier 1 — per-driver Δ + dominance-flip + all-PFT mosaic
    # ==================================================================

    def tier1_per_driver_panels(
        self, pft: str, phase: str, delta_table: Optional[pd.DataFrame] = None
    ) -> Optional[Path]:
        """3-panel figure: ΔTemp / ΔSol / ΔWater with significance stippling."""
        if delta_table is None:
            delta_table = self.an.epoch_delta(pft, phase, with_ci=True)
        if delta_table.empty:
            logger.warning(
                f"[{pft}/{phase}] no Δ data; skipping Tier 1 per-driver panel."
            )
            return None

        vmax = self.cfg.diverging_vmax
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig = plt.figure(figsize=(22, 8), facecolor="white")
        for i, g in enumerate(GROUPS, start=1):
            ax = fig.add_subplot(1, 3, i, projection=self._proj())
            self._setup_ax(
                ax,
                f"Δ{GROUP_LABELS[g]} ({self.cfg.late.label} − {self.cfg.early.label})",
            )
            data_2d = self._grid_2d(delta_table, f"d{g}")
            self._imshow_continuous(ax, data_2d, norm, self.DELTA_CMAP)
            sig_2d = self._grid_2d(delta_table, f"d{g}_sig").astype(bool)
            self._stipple(ax, sig_2d)

            sm = plt.cm.ScalarMappable(cmap=self.DELTA_CMAP, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(
                sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.7
            )
            cbar.set_label("Δ weight (Tmin+Tmax / Rad+Photo / Precip+SM)")

        fig.suptitle(
            f"{pft} {phase} — per-driver sensitivity shift, stipple = 95% CI excludes 0",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        path = self.out / f"{pft}_{phase}_delta_per_driver.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    def tier1_dominance_flip(
        self, pft: str, phase: str, delta_table: Optional[pd.DataFrame] = None
    ) -> Optional[Path]:
        """Categorical 3×3 transition map of argmax driver per epoch."""
        if delta_table is None:
            delta_table = self.an.epoch_delta(pft, phase, with_ci=False)
        if delta_table.empty:
            return None

        coded = self.an.dominance_codes(delta_table)
        codes_2d = self._grid_2d(coded, "flip_code")

        # Build categorical color image
        rgba = np.full((*codes_2d.shape, 4), [1.0, 1.0, 1.0, 1.0])
        for code, color in enumerate(self.FLIP_PALETTE):
            r, g, b = mcolors.to_rgb(color)
            mask = codes_2d == code
            rgba[mask] = [r, g, b, 1.0]

        lons_1d = np.linspace(-180.0, 180.0, 1440, endpoint=False)
        rgba_cyc, lon_cyc = add_cyclic_point(rgba, coord=lons_1d, axis=1)
        H, Wc = rgba_cyc.shape[:2]
        lats_col = np.linspace(90.0, -90.0, H)[:, None]
        alpha_mask = np.repeat(
            (lats_col >= self.cfg.crop_lat).astype(float), Wc, axis=1
        )
        rgba_cyc[..., 3] *= alpha_mask

        fig = plt.figure(figsize=(18, 9), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=self._proj())
        self._setup_ax(
            ax,
            f"{pft} {phase} — dominant driver flip ({self.cfg.early.label} → {self.cfg.late.label})",
        )
        ax.imshow(
            rgba_cyc,
            extent=[lon_cyc.min(), lon_cyc.max(), -90.0, 90.0],
            origin="upper",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
        )
        self._crop_extent(ax, lon_cyc.min(), lon_cyc.max())

        handles = [
            Patch(color=self.FLIP_PALETTE[i], label=self.FLIP_LABELS[i])
            for i in range(9)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=9,
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, -0.02),
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        path = self.out / f"{pft}_{phase}_dominance_flip.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    @staticmethod
    def _coarsen_delta(table: pd.DataFrame, resolution: float) -> pd.DataFrame:
        """Median-aggregate a Δ table to coarser grid cells.

        Uses floor-based binning so each bin [n, n+resolution) contains exactly
        (resolution/0.25)² quarter-degree sub-cells. Returns one row per coarse
        cell, with lat/lon set to the bin floor (e.g. lat=44.0 for the bin that
        covers 44.0, 44.25, 44.5, 44.75 when resolution=1.0).
        Bool columns ending in ``_sig`` use majority vote (>50% significant).
        """
        if table.empty or resolution <= 0.25:
            return table

        t = table.copy()
        t["_lat_bin"] = np.floor(t["latitude"] / resolution) * resolution
        t["_lon_bin"] = np.floor(t["longitude"] / resolution) * resolution

        skip = {"latitude", "longitude", "_lat_bin", "_lon_bin"}
        sig_cols = [c for c in t.columns if c not in skip and c.endswith("_sig")]
        num_cols = [
            c
            for c in t.columns
            if c not in skip
            and c not in sig_cols
            and pd.api.types.is_numeric_dtype(t[c])
        ]

        agg = (
            t.groupby(["_lat_bin", "_lon_bin"], as_index=False)[num_cols]
            .median()
            .rename(columns={"_lat_bin": "latitude", "_lon_bin": "longitude"})
        )

        for sc in sig_cols:
            frac = (
                t.groupby(["_lat_bin", "_lon_bin"])[sc]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "_lat_bin": "latitude",
                        "_lon_bin": "longitude",
                        sc: sc + "__frac",
                    }
                )
            )
            agg = agg.merge(frac, on=["latitude", "longitude"], how="left")
            agg[sc] = agg[sc + "__frac"] > 0.5
            agg.drop(columns=[sc + "__frac"], inplace=True)

        return agg

    @staticmethod
    def _expand_coarse_to_quarter(
        table: pd.DataFrame, resolution: float
    ) -> pd.DataFrame:
        """Expand coarse-grid rows back to all 0.25° sub-cells they cover.

        After ``_coarsen_delta`` each row's lat/lon is the bin floor.  This
        method replicates each row for every 0.25° sub-cell within the bin so
        that the existing ``_grid_2d`` machinery (which merges onto the 720×1440
        grid) fills solid colour blocks instead of single pixels.
        """
        if table.empty or resolution <= 0.25:
            return table

        n_sub = int(round(resolution / 0.25))
        offsets = np.array([i * 0.25 for i in range(n_sub)])
        dlat, dlon = np.meshgrid(offsets, offsets, indexing="ij")
        dlat = dlat.ravel()  # n_sub² values
        dlon = dlon.ravel()
        n_expand = len(dlat)

        rep = table.loc[table.index.repeat(n_expand)].reset_index(drop=True)
        tile_dlat = np.tile(dlat, len(table))
        tile_dlon = np.tile(dlon, len(table))
        rep["latitude"] = np.round(rep["latitude"].values + tile_dlat, 6)
        rep["longitude"] = np.round(rep["longitude"].values + tile_dlon, 6)
        return rep

    @staticmethod
    def _idw_fill(
        df: pd.DataFrame, radius: float, native_res: float = 0.25
    ) -> pd.DataFrame:
        """Fill empty grid cells within ``radius`` degrees using IDW.

        ``native_res`` sets the spacing of candidate target cells (default 0.25°
        for the raw quarter-degree data; pass the coarsen resolution when
        filling after aggregation).

        Weights: w_i = 1/d_i².  Boolean ``_sig`` columns are set to False for
        filled cells (they are interpolated, not bootstrapped).
        Fully vectorised — no Python loop over fill points.
        """
        if df.empty or radius <= 0:
            return df

        coarse_df = df
        res = native_res

        skip = {"latitude", "longitude"}
        sig_cols = [
            c for c in coarse_df.columns if c not in skip and c.endswith("_sig")
        ]
        num_cols = [
            c
            for c in coarse_df.columns
            if c not in skip
            and c not in sig_cols
            and pd.api.types.is_numeric_dtype(coarse_df[c])
        ]

        src_lats = coarse_df["latitude"].values.astype(np.float64)
        src_lons = coarse_df["longitude"].values.astype(np.float64)
        src_pts = np.column_stack([src_lats, src_lons])
        src_vals = coarse_df[num_cols].values.astype(np.float64)  # (n_src, n_cols)

        tree = cKDTree(src_pts)

        # Candidate target cells: 1-deg (or coarsen-res) grid spanning
        # the data extent plus one radius of padding on each side.
        lat_min = np.floor(src_lats.min() - radius)
        lat_max = np.ceil(src_lats.max() + radius)
        lon_min = np.floor(src_lons.min() - radius)
        lon_max = np.ceil(src_lons.max() + radius)

        tgt_lats = np.arange(max(-90.0, lat_min), min(90.0, lat_max + res), res)
        tgt_lons = np.arange(max(-180.0, lon_min), min(180.0, lon_max + res), res)
        TLat, TLon = np.meshgrid(tgt_lats, tgt_lons, indexing="ij")
        all_tgt = np.column_stack([TLat.ravel(), TLon.ravel()])

        # Drop targets that already have a real data cell.
        existing = set(zip(np.round(src_lats, 4), np.round(src_lons, 4)))
        mask_empty = np.array(
            [(round(r[0], 4), round(r[1], 4)) not in existing for r in all_tgt]
        )
        tgt_empty = all_tgt[mask_empty]

        if len(tgt_empty) == 0:
            return coarse_df

        # For each empty target, find source indices within radius.
        neighbor_lists = tree.query_ball_point(tgt_empty, r=radius)

        # Keep only targets that actually have neighbours.
        has_nbr = np.array([len(nb) > 0 for nb in neighbor_lists], dtype=bool)
        tgt_fill = tgt_empty[has_nbr]
        nbr_lists = [nb for nb, h in zip(neighbor_lists, has_nbr) if h]

        if len(tgt_fill) == 0:
            return coarse_df

        n_fill = len(tgt_fill)
        logger.debug(f"IDW fill: {n_fill} cells from {len(coarse_df)} source cells")

        # Vectorised IDW: flatten all (fill_point, source) pairs.
        tgt_idx = np.concatenate(
            [np.full(len(nb), i, dtype=np.int32) for i, nb in enumerate(nbr_lists)]
        )
        src_idx = np.concatenate([np.array(nb, dtype=np.int32) for nb in nbr_lists])

        diffs = src_pts[src_idx] - tgt_fill[tgt_idx]  # (n_pairs, 2)
        dists = np.maximum(np.linalg.norm(diffs, axis=1), 1e-6)
        weights = 1.0 / dists**2  # (n_pairs,)

        # Weighted sum per fill point for each numeric column.
        num_out = np.full((n_fill, len(num_cols)), np.nan, dtype=np.float64)
        w_sum = np.zeros(n_fill, dtype=np.float64)
        np.add.at(w_sum, tgt_idx, weights)
        for j in range(len(num_cols)):
            wv = weights * src_vals[src_idx, j]
            col_out = np.zeros(n_fill, dtype=np.float64)
            np.add.at(col_out, tgt_idx, wv)
            num_out[:, j] = np.where(w_sum > 0, col_out / w_sum, np.nan)

        fill_df = pd.DataFrame(
            np.column_stack([tgt_fill, num_out]),
            columns=["latitude", "longitude"] + num_cols,
        )
        for sc in sig_cols:
            fill_df[sc] = False

        # Align column order with source table.
        for col in coarse_df.columns:
            if col not in fill_df.columns:
                fill_df[col] = np.nan
        fill_df = fill_df[coarse_df.columns]

        return pd.concat([coarse_df, fill_df], ignore_index=True)

    def _coarsen_for_plot(
        self, table: pd.DataFrame, apply_idw: bool = True
    ) -> pd.DataFrame:
        """Remove isolated outliers, optionally IDW-fill, coarsen, clip, and re-expand.

        ``apply_idw=False`` skips the IDW gap-fill step (used for EOS maps where
        interpolation is not desired).
        """
        res = self.cfg.coarsen_resolution
        if res <= 0.25:
            return self._remove_isolated(table)
        # 1. Remove stray isolated pixels (Sahara / ocean artifacts)
        cleaned = self._remove_isolated(table)
        # 2. IDW gap-fill at native 0.25° (SOS only)
        filled = (
            self._idw_fill(cleaned, self.cfg.idw_radius, native_res=0.25)
            if apply_idw and self.cfg.idw_radius > 0
            else cleaned
        )
        # 3. Aggregate to coarse grid
        coarsened = self._coarsen_delta(filled, res)
        # 4. Clip extreme delta outliers at the coarse level
        clipped = self._clip_outliers(coarsened)
        # 5. Expand back to 0.25° sub-cell blocks for _grid_2d
        return self._expand_coarse_to_quarter(clipped, res)

    @staticmethod
    def _data_driven_vmax(
        series: pd.Series, clip: float = 0.15, percentile: float = 98.0
    ) -> float:
        """98th-percentile of |Δ|, capped at clip. Ensures small deltas fill the colorbar."""
        vals = series.dropna().to_numpy()
        if len(vals) == 0:
            return clip
        return float(min(clip, max(1e-4, np.percentile(np.abs(vals), percentile))))

    @staticmethod
    def _remove_isolated(
        df: pd.DataFrame, min_neighbors: int = 1, radius: float = 3.5
    ) -> pd.DataFrame:
        """Drop rows with fewer than min_neighbors other rows within radius degrees.

        Catches truly isolated stray pixels (e.g. single points in the Sahara or
        open ocean) that are artifacts of the biome detection rather than real
        phenology signals.  The default (1 neighbor within 3.5°) is intentionally
        lenient — it only removes completely isolated single pixels.
        """
        if df.empty or len(df) <= min_neighbors:
            return df
        pts = df[["latitude", "longitude"]].values.astype(np.float64)
        tree = cKDTree(pts)
        # query_ball_point returns the point itself too, so subtract 1
        counts = np.array([len(nb) - 1 for nb in tree.query_ball_point(pts, r=radius)])
        return df[counts >= min_neighbors].reset_index(drop=True)

    @staticmethod
    def _clip_outliers(df: pd.DataFrame, iqr_factor: float = 3.0) -> pd.DataFrame:
        """Clip extreme delta values that lie beyond median ± iqr_factor × IQR.

        Applied per driver group (dtemp, dsol, dp) so that a handful of extreme
        pixels don't bleed color into the colorbar or appear as bright artifacts
        on the map.
        """
        if df.empty:
            return df
        out = df.copy()
        for g in GROUPS:
            col = f"d{g}"
            if col not in out.columns:
                continue
            vals = out[col].dropna()
            if len(vals) < 4:
                continue
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            lo = q25 - iqr_factor * iqr
            hi = q75 + iqr_factor * iqr
            out[col] = out[col].clip(lower=lo, upper=hi)
        return out

    def tier1_mosaic(self, phase: str) -> List[Path]:
        """All-PFT Δ mosaic — three per-driver maps (data-driven colorbar) + dominance-flip."""
        mosaic = self.an.mosaic_delta(phase, with_ci=True)
        if mosaic.empty:
            logger.warning(f"All-PFT mosaic empty for {phase}")
            return []

        # Coarsen for rendering (fills sparse PFT biome gaps).
        # IDW fill is applied for SOS only; EOS uses no interpolation.
        # vmax is computed from the original data so aggregation doesn't alter the scale.
        plot_mosaic = self._coarsen_for_plot(mosaic, apply_idw=(phase == "SOS"))
        res = self.cfg.coarsen_resolution
        res_label = f"{res:g}°" if res > 0.25 else "0.25°"

        paths: List[Path] = []

        # Three per-driver mosaic maps — each gets its own data-driven colorbar
        for g in GROUPS:
            vmax = self._data_driven_vmax(mosaic[f"d{g}"])
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            fig = plt.figure(figsize=(18, 9), facecolor="white")
            ax = fig.add_subplot(1, 1, 1, projection=self._proj())
            self._setup_ax(
                ax,
                f"All-PFT mosaic — Δ{GROUP_LABELS[g]}  {phase}  "
                f"({self.cfg.late.label} − {self.cfg.early.label})",
            )
            data_2d = self._grid_2d(plot_mosaic, f"d{g}")
            self._imshow_continuous(ax, data_2d, norm, self.DELTA_CMAP)
            sm = plt.cm.ScalarMappable(cmap=self.DELTA_CMAP, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(
                sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.7
            )
            cbar.set_label(f"Δ weight  [±{vmax:.3f}]  |  grid: {res_label}")
            plt.tight_layout()
            path = self.out / f"AllPFT_{phase}_delta_{g}.png"
            plt.savefig(
                path,
                dpi=self.cfg.dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            plt.close(fig)
            paths.append(path)
            logger.info(f"Saved {path}")

        # Mosaic dominance-flip — coarsen + expand so blocks match the delta maps
        coded_coarse = self.an.dominance_codes(
            self._coarsen_delta(mosaic, res) if res > 0.25 else mosaic
        )
        coded_plot = (
            self._expand_coarse_to_quarter(coded_coarse, res)
            if res > 0.25
            else coded_coarse
        )
        codes_2d = self._grid_2d(coded_plot, "flip_code")
        rgba = np.full((*codes_2d.shape, 4), [1.0, 1.0, 1.0, 1.0])
        for code, color in enumerate(self.FLIP_PALETTE):
            r, gg, b = mcolors.to_rgb(color)
            rgba[codes_2d == code] = [r, gg, b, 1.0]

        lons_1d = np.linspace(-180.0, 180.0, 1440, endpoint=False)
        rgba_cyc, lon_cyc = add_cyclic_point(rgba, coord=lons_1d, axis=1)
        H, Wc = rgba_cyc.shape[:2]
        lats_col = np.linspace(90.0, -90.0, H)[:, None]
        alpha_mask = np.repeat(
            (lats_col >= self.cfg.crop_lat).astype(float), Wc, axis=1
        )
        rgba_cyc[..., 3] *= alpha_mask

        fig = plt.figure(figsize=(18, 9), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=self._proj())
        self._setup_ax(
            ax,
            f"All-PFT mosaic — dominance flip  {phase}  "
            f"({self.cfg.early.label} → {self.cfg.late.label})",
        )
        ax.imshow(
            rgba_cyc,
            extent=[lon_cyc.min(), lon_cyc.max(), -90.0, 90.0],
            origin="upper",
            transform=ccrs.PlateCarree(),
            interpolation="nearest",
        )
        self._crop_extent(ax, lon_cyc.min(), lon_cyc.max())
        handles = [
            Patch(color=self.FLIP_PALETTE[i], label=self.FLIP_LABELS[i])
            for i in range(9)
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=9,
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(0.5, -0.02),
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
        path = self.out / f"AllPFT_{phase}_dominance_flip.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        paths.append(path)
        logger.info(f"Saved {path}")
        return paths

    def tier1_combined_delta_panel(self) -> Optional[Path]:
        """2×3 figure: rows = SOS / EOS, columns = temp / sol / p.

        SOS row: fixed shared colorbar ±0.15 (IDW fill applied).
        EOS row: fixed shared colorbar ±0.025 (no IDW fill).
        Each row shares one horizontal colorbar spanning all three panels.
        Isolated outlier pixels and extreme delta values are removed before rendering.
        """
        mosaics: Dict[str, pd.DataFrame] = {}
        for phase in ("SOS", "EOS"):
            m = self.an.mosaic_delta(phase, with_ci=True)
            if m.empty:
                logger.warning(f"Combined panel: mosaic empty for {phase}; skipping.")
                return None
            mosaics[phase] = m

        phases = ("SOS", "EOS")
        groups = ("temp", "sol", "p")
        group_titles = {
            "temp": "Δ Temperature (Tmin+Tmax)",
            "sol": "Δ Solar (Rad+Photo)",
            "p": "Δ Water (Precip+SM)",
        }
        # Fixed colorbar range per phase row
        phase_vmax = {"SOS": 0.15, "EOS": 0.025}

        res = self.cfg.coarsen_resolution
        res_label = f"{res:g}°" if res > 0.25 else "0.25°"

        # Coarsen + clean each mosaic (IDW only for SOS)
        plot_mosaics: Dict[str, pd.DataFrame] = {
            ph: self._coarsen_for_plot(m, apply_idw=(ph == "SOS"))
            for ph, m in mosaics.items()
        }

        # GridSpec: 2 rows × (3 map cols + 1 narrow colorbar col)
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(26, 14), facecolor="white")
        gs = GridSpec(
            2, 4, figure=fig, width_ratios=[1, 1, 1, 0.04], wspace=0.08, hspace=0.15
        )

        for row, phase in enumerate(phases):
            vmax = phase_vmax[phase]
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            plot_mosaic = plot_mosaics[phase]
            for col, g in enumerate(groups):
                ax = fig.add_subplot(gs[row, col], projection=self._proj())
                self._setup_ax(ax, f"{phase} — {group_titles[g]}")
                data_2d = self._grid_2d(plot_mosaic, f"d{g}")
                self._imshow_continuous(ax, data_2d, norm, self.DELTA_CMAP)

            # Dedicated colorbar axes in the 4th column of this row
            cax = fig.add_subplot(gs[row, 3])
            sm = plt.cm.ScalarMappable(cmap=self.DELTA_CMAP, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
            cbar.set_label(f"{phase} Δ weight  [±{vmax:.3f}]", fontsize=10)

        fig.suptitle(
            f"All-PFT Δ sensitivity  ({self.cfg.late.label} − {self.cfg.early.label})"
            f"  |  {res_label} grid",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )
        path = self.out / "AllPFT_combined_delta_panel.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    # ==================================================================
    # Tier 2 — zonal profiles, summary heatmap, |Δ| magnitude
    # ==================================================================

    def tier2_zonal_profile(
        self,
        pft: str,
        phase: str,
        delta_table: Optional[pd.DataFrame] = None,
        lat_step: float = 1.0,
    ) -> Optional[Path]:
        if delta_table is None:
            delta_table = self.an.epoch_delta(pft, phase, with_ci=False)
        profile = self.an.zonal_profile(pft, phase, delta_table, lat_step=lat_step)
        if profile.empty:
            return None

        fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
        for g in GROUPS:
            color = GROUP_COLORS[g]
            ax.plot(
                profile[f"d{g}_median"],
                profile["lat_bin"],
                color=color,
                lw=2,
                label=f"Δ{g}",
            )
            ax.fill_betweenx(
                profile["lat_bin"],
                profile[f"d{g}_q25"],
                profile[f"d{g}_q75"],
                color=color,
                alpha=0.15,
            )
        ax.axvline(0.0, color="black", lw=0.5, linestyle="--")
        ax.set_xlabel(f"Δ weight ({self.cfg.late.label} − {self.cfg.early.label})")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{pft} {phase} — zonal Δsensitivity profile")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        plt.tight_layout()
        path = self.out / f"{pft}_{phase}_zonal_profile.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    def tier2_summary_heatmap(self, summary: pd.DataFrame) -> Optional[Path]:
        if summary.empty:
            return None

        summary = summary.copy()
        summary["row"] = summary["pft"] + " " + summary["phase"]
        pivot = summary.set_index("row")[[f"mean_d{g}" for g in GROUPS]]
        pivot.columns = [GROUP_LABELS[g] for g in GROUPS]

        sig_pos = summary.set_index("row")[[f"frac_sig_pos_d{g}" for g in GROUPS]]
        sig_neg = summary.set_index("row")[[f"frac_sig_neg_d{g}" for g in GROUPS]]

        vmax = max(0.05, float(np.nanmax(np.abs(pivot.values))))
        fig, ax = plt.subplots(
            figsize=(8.5, max(4, 0.55 * len(pivot))), facecolor="white"
        )
        im = ax.imshow(
            pivot.values,
            cmap=self.DELTA_CMAP,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            aspect="auto",
        )

        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j, g in enumerate(GROUPS):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                pos = sig_pos.values[i, j]
                neg = sig_neg.values[i, j]
                pos_str = f"+{pos:.0%}" if not np.isnan(pos) else ""
                neg_str = f"−{neg:.0%}" if not np.isnan(neg) else ""
                ax.text(
                    j,
                    i,
                    f"{v:+.3f}\n{pos_str} / {neg_str}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(v) < vmax * 0.6 else "white",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("Area-weighted mean Δ weight")
        ax.set_title(
            "PFT × phase × driver  —  mean Δ  and  fraction of pixels significant"
        )
        plt.tight_layout()
        path = self.out / "AllPFT_summary_heatmap.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    def tier2_magnitude(
        self, pft: str, phase: str, delta_table: Optional[pd.DataFrame] = None
    ) -> Optional[Path]:
        """|Δ| per driver — highlights hotspots regardless of sign."""
        if delta_table is None:
            delta_table = self.an.epoch_delta(pft, phase, with_ci=False)
        if delta_table.empty:
            return None
        vmax = self.cfg.diverging_vmax
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        fig = plt.figure(figsize=(22, 8), facecolor="white")
        for i, g in enumerate(GROUPS, start=1):
            ax = fig.add_subplot(1, 3, i, projection=self._proj())
            self._setup_ax(ax, f"|Δ{GROUP_LABELS[g]}|  {pft} {phase}")
            data_2d = np.abs(self._grid_2d(delta_table, f"d{g}"))
            self._imshow_continuous(ax, data_2d, norm, "magma")
            sm = plt.cm.ScalarMappable(cmap="magma", norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(
                sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.7
            )
            cbar.set_label("|Δ weight|")
        plt.tight_layout()
        path = self.out / f"{pft}_{phase}_magnitude.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    # ==================================================================
    # Tier 3 — dominance margin Δ + decadal trajectory
    # ==================================================================

    def tier3_dominance_margin(
        self, pft: str, phase: str, delta_table: Optional[pd.DataFrame] = None
    ) -> Optional[Path]:
        if delta_table is None:
            delta_table = self.an.epoch_delta(pft, phase, with_ci=False)
        if delta_table.empty:
            return None
        coded = self.an.dominance_codes(delta_table)
        vmax = max(0.05, float(np.nanmax(np.abs(coded["dmargin"]))))
        vmax = min(vmax, 0.25)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig = plt.figure(figsize=(18, 9), facecolor="white")
        ax = fig.add_subplot(1, 1, 1, projection=self._proj())
        self._setup_ax(
            ax,
            f"{pft} {phase} — Δdominance margin "
            f"({self.cfg.late.label} − {self.cfg.early.label}). "
            "Red = top driver more dominant; blue = less.",
        )
        data_2d = self._grid_2d(coded, "dmargin")
        self._imshow_continuous(ax, data_2d, norm, "PuOr_r")
        sm = plt.cm.ScalarMappable(cmap="PuOr_r", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(
            sm, ax=ax, orientation="horizontal", pad=0.04, fraction=0.04, shrink=0.7
        )
        cbar.set_label("Δ(top − second) weight")
        plt.tight_layout()
        path = self.out / f"{pft}_{phase}_dominance_margin.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path

    def tier3_decadal_trajectory(
        self, pft: str, phase: str, lat_step: float = 5.0
    ) -> Optional[Path]:
        rolling = self.an.rolling_zonal(pft, phase, lat_step=lat_step)
        if rolling.empty:
            return None

        # Heatmap per driver: y = lat, x = window center, color = weight
        fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="white", sharey=True)
        for ax, g in zip(axes, GROUPS):
            sub = rolling[rolling["group"] == g]
            pivot = sub.pivot(index="lat_bin", columns="window_center", values="value")
            pivot = pivot.sort_index(ascending=True)
            im = ax.imshow(
                pivot.values,
                aspect="auto",
                cmap="viridis",
                origin="lower",
                extent=[
                    pivot.columns.min(),
                    pivot.columns.max(),
                    pivot.index.min(),
                    pivot.index.max(),
                ],
            )
            ax.set_title(
                f"{g}: rolling {self.cfg.rolling_width}-yr zonal-median weight"
            )
            ax.set_xlabel("Window centre (year)")
            ax.grid(False)
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
            cbar.set_label("Median weight")
        axes[0].set_ylabel("Latitude")
        fig.suptitle(
            f"{pft} {phase} — decadal trajectory of zonal-mean sensitivity",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = self.out / f"{pft}_{phase}_decadal_trajectory.png"
        plt.savefig(
            path,
            dpi=self.cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path
