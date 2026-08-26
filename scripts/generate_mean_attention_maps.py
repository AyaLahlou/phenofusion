#!/usr/bin/env python3
"""
Generate Attention Mean Day Maps for Phenological Analysis

Creates global maps showing the attention-weighted mean day of the 365-day
historical time axis, per plant functional type (PFT) and phenological
phase (SOS/EOS).

Unlike the max-attention-window pipeline (which finds a 30-day window's
start index), this metric is a continuous expected value over every day of
the historical axis: each pixel's color represents how far back, on
average, the model's attention is centered (0 = attention centered on the
oldest historical day, ~365 = attention centered right at the forecast
horizon).

Input CSVs are produced by:
    scripts/get_attention_mean_data.py

Schema expected:  location | latitude | longitude | mean_attention_day

This is a thin CLI wrapper around the same `WindowMapGenerator` used by
`generate_window_maps.py`, configured for the mean-day metric
(`value_col="mean_attention_day", legacy_offset=365`).
"""

import logging

from generate_window_maps import WindowMapGenerator, create_file_mapping

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate global maps of attention-weighted mean day indices"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing mean_attention_day CSV files",
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
    parser.add_argument(
        "--crop-lat-min",
        type=float,
        default=-60,
        help="Hide latitudes below this value (default: -60)",
    )
    parser.add_argument(
        "--crop-lat-max",
        type=float,
        default=71,
        help="Hide latitudes above this value (default: 71)",
    )
    parser.add_argument(
        "--crop-lon-min",
        type=float,
        default=-160,
        help="West edge of the view (default: -160, tight to Alaska)",
    )
    parser.add_argument(
        "--crop-lon-max",
        type=float,
        default=165,
        help="East edge of the view (default: 165, tight to New Zealand)",
    )

    args = parser.parse_args()

    gen = WindowMapGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        value_col="mean_attention_day",
        legacy_offset=365.0,
    )

    file_mapping = create_file_mapping(args.data_dir)

    plot_kwargs = dict(
        projection=args.projection,
        crop_lat=args.crop_lat_min,
        crop_lat_max=args.crop_lat_max,
        crop_lon_min=args.crop_lon_min,
        crop_lon_max=args.crop_lon_max,
        dpi=args.dpi,
        show=args.show_plots,
        colorbar_label="Attention-weighted memory depth (days before forecast horizon)",
    )

    # Individual PFT maps
    for pft in ["BET", "NET", "NDT", "SHR", "GRA"]:
        for phase in ["SOS", "EOS"]:
            if pft in file_mapping and phase in file_mapping[pft]:
                gen.generate_pft_map(
                    [file_mapping[pft][phase]],
                    [pft],
                    f"{pft}_MeanAttentionDepth_{phase}",
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
                f"BDT_MeanAttentionDepth_{phase}",
                **plot_kwargs,
            )
        elif "BDT" in file_mapping and phase in file_mapping["BDT"]:
            gen.generate_pft_map(
                [file_mapping["BDT"][phase]],
                ["BDT"],
                f"BDT_MeanAttentionDepth_{phase}",
                **plot_kwargs,
            )

    # All-PFTs combined map. BDT (and its latitude-band variants) is drawn
    # first as the background layer, since it spans nearly the whole globe
    # and would otherwise blend with / wash out sparser PFTs (e.g. NET)
    # wherever their footprints overlap. Other PFTs are stacked on top.
    for phase in ["SOS", "EOS"]:
        bdt_files, bdt_names, other_files, other_names = [], [], [], []
        for pft, phases in file_mapping.items():
            if phase not in phases:
                continue
            if pft.startswith("BDT"):
                bdt_files.append(phases[phase])
                bdt_names.append(pft)
            else:
                other_files.append(phases[phase])
                other_names.append(pft)
        all_files = bdt_files + other_files
        all_names = bdt_names + other_names
        if all_files:
            gen.generate_pft_map(
                all_files,
                all_names,
                f"All_PFTs_MeanAttentionDepth_{phase}",
                **plot_kwargs,
            )

    # Multi-panel map (one subplot per PFT, shared colorbar)
    for phase in ["SOS", "EOS"]:
        gen.generate_multi_panel_map(
            file_mapping,
            phase,
            projection=args.projection,
            crop_lat=args.crop_lat_min,
            crop_lat_max=args.crop_lat_max,
            crop_lon_min=args.crop_lon_min,
            crop_lon_max=args.crop_lon_max,
            dpi=args.dpi,
            show=args.show_plots,
        )

    logger.info("Mean attention day map generation complete.")


if __name__ == "__main__":
    main()
