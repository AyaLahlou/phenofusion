#!/usr/bin/env python3
"""
Stage 3 runner: produce all Tier 1/2/3 sensitivity-shift figures.

Workflow:
    1. Auto-discover `{PFT}_yearly_drivers*.csv` in Data/data_output/sensitivity_data/.
    2. For each (PFT, phase):
         - compute the Δ table + bootstrap CIs once,
         - feed the cached table into every Tier 1/2/3 builder.
    3. Build the all-PFT Δ mosaic (Tier 1) once per phase.
    4. Build the PFT × phase × driver summary heatmap (Tier 2) at the end.

Run from the repo root:
    python Figures/figures_wrangling/run_sensitivity_shifts.py

Outputs land in Figures/figures_output/sensitivity/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when invoked as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phenofusion.sensitivity_shifts import (  # noqa: E402
    Epoch,
    SensitivityShiftAnalyzer,
    SensitivityShiftPlotter,
    ShiftConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PHASES = ("SOS", "EOS")


def _discover_pfts(data_dir: Path, requested: list[str] | None) -> list[str]:
    found = sorted(
        {
            p.name.split("_yearly_drivers")[0]
            for p in data_dir.glob("*_yearly_drivers*.csv")
        }
    )
    if not found:
        logger.error(
            f"No yearly driver CSVs in {data_dir}. "
            "Run Data/data_wrangling/extract_epoch_drivers.py on HPC first."
        )
        return []
    if requested:
        missing = [p for p in requested if p not in found]
        if missing:
            logger.warning(f"Requested PFTs missing on disk: {missing}")
        return [p for p in requested if p in found]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity-shift figure runner.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sensitivity_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sensitivity_figures",
    )
    parser.add_argument("--early-start", type=int, default=1982)
    parser.add_argument("--early-end", type=int, default=1991)
    parser.add_argument("--late-start", type=int, default=2012)
    parser.add_argument("--late-end", type=int, default=2021)
    parser.add_argument(
        "--legacy-sol",
        action="store_true",
        help="Use 2*hist_rad for the solar composite (snapshot bug-compat).",
    )
    parser.add_argument("--bootstrap-B", type=int, default=200)
    parser.add_argument(
        "--projection", choices=["Robinson", "PlateCarree"], default="Robinson"
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pfts", nargs="*", help="Override auto-discovered PFTs.")
    parser.add_argument(
        "--skip-tier",
        nargs="*",
        default=[],
        choices=["1", "2", "3", "mosaic", "summary"],
        help="Skip selected tier or aggregate figures.",
    )
    parser.add_argument("--diverging-vmax", type=float, default=0.15)
    parser.add_argument(
        "--coarsen-resolution",
        type=float,
        default=1.0,
        help="Aggregate mosaic maps to this grid resolution (degrees) before "
        "rendering. 1.0 = 1° blocks (default). 0.25 = no aggregation.",
    )
    parser.add_argument(
        "--idw-radius",
        type=float,
        default=1.1,
        help="IDW fill radius (degrees) applied after coarsening. "
        "Empty cells within this distance of real data are interpolated. "
        "0 = disable IDW fill.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return 1

    config = ShiftConfig(
        early=Epoch(
            f"{args.early_start}-{args.early_end}", args.early_start, args.early_end
        ),
        late=Epoch(
            f"{args.late_start}-{args.late_end}", args.late_start, args.late_end
        ),
        legacy_sol=args.legacy_sol,
        bootstrap_B=args.bootstrap_B,
        projection=args.projection,
        dpi=args.dpi,
        diverging_vmax=args.diverging_vmax,
        coarsen_resolution=args.coarsen_resolution,
        idw_radius=args.idw_radius,
    )

    analyzer = SensitivityShiftAnalyzer(args.data_dir, config)
    plotter = SensitivityShiftPlotter(analyzer, args.output_dir)

    pfts = _discover_pfts(args.data_dir, args.pfts)
    if not pfts:
        return 1
    logger.info(f"Discovered PFTs: {pfts}")

    summary_rows: list[dict] = []
    skip = set(args.skip_tier)

    for pft in pfts:
        for phase in PHASES:
            logger.info(f"==== {pft} / {phase} ====")
            try:
                delta = analyzer.epoch_delta(pft, phase, with_ci=True)
            except FileNotFoundError as e:
                logger.warning(str(e))
                continue
            if delta.empty:
                logger.warning(f"No Δ pixels for {pft}/{phase}; skipping.")
                continue

            if "1" not in skip:
                plotter.tier1_per_driver_panels(pft, phase, delta_table=delta)
                plotter.tier1_dominance_flip(pft, phase, delta_table=delta)
            if "2" not in skip:
                plotter.tier2_zonal_profile(pft, phase, delta_table=delta)
                plotter.tier2_magnitude(pft, phase, delta_table=delta)
            if "3" not in skip:
                plotter.tier3_dominance_margin(pft, phase, delta_table=delta)
                plotter.tier3_decadal_trajectory(pft, phase)

            summary_rows.append(analyzer.summary_row(pft, phase, delta))

    if "mosaic" not in skip:
        for phase in PHASES:
            logger.info(f"==== All-PFT mosaic / {phase} ====")
            plotter.tier1_mosaic(phase)
        logger.info("==== All-PFT combined delta panel (2x3) ====")
        plotter.tier1_combined_delta_panel()

    if "summary" not in skip and "2" not in skip and summary_rows:
        logger.info("==== Summary heatmap ====")
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(args.output_dir / "summary_table.csv", index=False)
        plotter.tier2_summary_heatmap(summary_df)

    logger.info("Sensitivity-shift figure generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
