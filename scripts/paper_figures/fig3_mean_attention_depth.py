#!/usr/bin/env python3
"""Figure 3 (mean-day variant) | Attention-weighted memory depth.

Same visual design as `fig3_legacy_depth.py`, but the underlying per-pixel
metric is the attention-weighted **mean day** of the 365-day historical
axis (see `scripts/get_attention_mean_data.py`) rather than the start of
the single 30-day window of maximum attention. Every historical timestep
contributes to the mean in proportion to its attention weight, so this is
a continuous "center of mass" rather than a hard argmax over windows.

"Memory depth" = 365 - mean_attention_day: how many days before the
forecast horizon the attention-weighted center of the TFT's historical
attention falls. ~1 = attention centered on the day right before the
forecast horizon; 365 = attention centered on the oldest historical day.

a,b  Global memory-depth maps, pooled across all six PFTs, side by side for
     start-of-season (a) and end-of-season (b) predictions. BDT (which
     covers nearly the whole globe) is drawn as the background layer so
     sparser PFTs -- e.g. NET -- are not blended away wherever their
     footprints overlap it.

Writes data/attention_mean/paper_figures/Fig3_mean_attention_depth.{png,pdf}
"""

import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import apply_style, panel_label  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from generate_window_maps import WindowMapGenerator, create_file_mapping  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data" / "attention_mean"
OUT_DIR = DATA_DIR / "paper_figures"

LAT_MIN, LAT_MAX = -58, 78
GRID_RES = 0.25
VMIN, VMAX = 0, 365
VALUE_COL = "mean_attention_day"
LEGACY_OFFSET = 365.0


# ----------------------------------------------------------------------
# Data assembly
# ----------------------------------------------------------------------


def _bdt_sources(file_mapping, phase):
    bands = ["BDT_5020", "BDT_2020", "BDT_2060", "BDT_5090"]
    files = [
        (file_mapping[b][phase], b) for b in bands if phase in file_mapping.get(b, {})
    ]
    if files:
        return files
    if phase in file_mapping.get("BDT", {}):
        return [(file_mapping["BDT"][phase], "BDT")]
    return []


def all_pft_grid(gen: WindowMapGenerator, file_mapping, phase: str) -> np.ndarray:
    """Reproduce the stacked All-PFTs grid (BDT background, others on top)."""
    bdt = _bdt_sources(file_mapping, phase)
    others = [
        (file_mapping[p][phase], p)
        for p in file_mapping
        if not p.startswith("BDT") and phase in file_mapping.get(p, {})
    ]
    dataframes = []
    for fp, name in bdt + others:
        df = gen.load_csv_data(fp, name)
        if not df.empty:
            dataframes.append(df)
    grid_df = gen.build_grid(dataframes)
    data_2d = np.reshape(grid_df["legacy_depth"].values, (720, 1440)).copy()
    return gen._apply_seam_row_fixes(data_2d)


# ----------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------


def build_edges(centers: np.ndarray, res: float) -> np.ndarray:
    c = np.sort(np.asarray(centers, dtype=float))
    return np.concatenate([c - res / 2, [c[-1] + res / 2]])


def add_season_ticks(cax, phase: str) -> None:
    """Label each quarter of the colorbar with the season it falls in."""
    labels = WindowMapGenerator.SEASON_LABELS[phase]
    edges = np.linspace(VMIN, VMAX, 5)
    for edge in edges[1:-1]:
        cax.axvline(edge, color="black", lw=0.4, alpha=0.45, ymin=0, ymax=1)
    centers = (edges[:-1] + edges[1:]) / 2
    for c, lab in zip(centers, labels):
        cax.text(
            c,
            1.55,
            lab,
            transform=cax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
            color="0.35",
        )


def draw_map(fig, ax, cax, data_2d, gen, phase, title):
    lat_edges = build_edges(gen.GRID_LATS, GRID_RES)
    lon_edges = build_edges(gen.GRID_LONS, GRID_RES)
    data_asc = np.flipud(data_2d)

    ax.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f2f2f0", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        np.ma.masked_invalid(data_asc),
        cmap=gen._resolve_cmap(),
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="flat",
        zorder=1,
        rasterized=True,
    )
    ax.coastlines(linewidth=0.25, color="#555555", zorder=2)
    ax.spines["geo"].set_linewidth(0.5)
    ax.spines["geo"].set_edgecolor("#999999")
    ax.set_title(title, pad=3, fontsize=8, fontweight="bold", color="black")

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(
        "Attention-weighted memory depth (days before forecast horizon)",
        labelpad=2,
        fontsize=6.5,
    )
    cbar.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=6, width=0.4, length=2)
    add_season_ticks(cax, phase)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


def main():
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gen = WindowMapGenerator(
        data_dir=str(DATA_DIR),
        output_dir=str(OUT_DIR),
        cmap="tab20_no_blue",
        value_col=VALUE_COL,
        legacy_offset=LEGACY_OFFSET,
    )
    file_mapping = create_file_mapping(str(DATA_DIR))
    grids = {phase: all_pft_grid(gen, file_mapping, phase) for phase in ("SOS", "EOS")}

    fig_w = 9.6
    left, right, gap = 0.14, 0.14, 0.16
    map_w = (fig_w - left - right - gap) / 2
    H = map_w / 2.414  # Robinson clipped to [-58, 78]: measured width:height ratio
    top_pad, cb_gap, cb_h, bot_pad = 0.24, 0.20, 0.09, 0.16
    fig_h = top_pad + H + cb_gap + cb_h + bot_pad
    fig = plt.figure(figsize=(fig_w, fig_h))

    def fy(inches):
        return inches / fig_h

    def fx(inches):
        return inches / fig_w

    proj = ccrs.Robinson()
    titles = {"SOS": "Start of Season", "EOS": "End of Season"}
    letters = {"SOS": "a", "EOS": "b"}
    cax_w = 0.70 * map_w
    for i, phase in enumerate(("SOS", "EOS")):
        x0 = left + i * (map_w + gap)
        ax = fig.add_axes(
            [fx(x0), 1 - fy(top_pad + H), fx(map_w), fy(H)], projection=proj
        )
        cax = fig.add_axes(
            [
                fx(x0 + (map_w - cax_w) / 2),
                1 - fy(top_pad + H + cb_gap + cb_h),
                fx(cax_w),
                fy(cb_h),
            ]
        )
        draw_map(fig, ax, cax, grids[phase], gen, phase, titles[phase])
        panel_label(ax, letters[phase], x=0.005, y=0.92)

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Fig3_mean_attention_depth.{ext}", dpi=450)
    print("wrote", OUT_DIR / "Fig3_mean_attention_depth.png")


if __name__ == "__main__":
    main()
