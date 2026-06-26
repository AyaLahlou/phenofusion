#!/usr/bin/env python3
"""
Epoch-stratified driver weight extraction.

Produces one CSV per PFT with one row per (location, phase, year), preserving
all sample-level driver weights. This single file feeds three downstream
analyses:

  - Fixed-epoch contrast (e.g. 1982-1991 vs 2012-2021)
  - Bootstrap resampling for significance
  - Rolling decadal-window trajectories

Output schema
-------------
    location, latitude, longitude, phase, year,
    hist_tmin, hist_tmax, hist_rad, hist_precip, hist_photo, hist_sm

Compared to `get_driver_map_data_refactored.py`, this script:
  1. Parses the date suffix in each test sample ID (`{location}_{ISO-date}`)
     to obtain the calendar year.
  2. Emits one row per sample (not a per-location median), so downstream
     code can stratify, bootstrap, or roll windows freely.
  3. Reuses the canonical sliding 30-day max-attention window logic and the
     median-within-window summarisation that the rest of the pipeline uses.

SOS/EOS index detection
-----------------------
Two modes are supported:

  a) Pre-computed index files (``--sos-indices`` / ``--eos-indices``):
     Pass numpy arrays of prediction-array indices directly. Use this if
     such files already exist from a previous pipeline run.

  b) CSIF-slope detection (``--detect-csif``):
     Derives phenological transitions from the CSIF time series embedded
     in the test set, replicating the method in archive/generate_sensitivity_plots.py:
     - Slide a 30-sample window over each (location, time) sequence.
     - If |CSIF[0] − CSIF[−1]| > min_diff and slope > threshold → SOS.
     - If slope < −threshold → EOS.
     This mode requires no external index files and is the recommended
     fallback when index files do not exist.

Run on HPC where the prediction pickles live. Resulting CSVs are small enough
to rsync back to the repo's `Data/data_output/sensitivity_data/`.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DRIVER_NAMES: Tuple[str, ...] = ("tmin", "tmax", "rad", "precip", "photo", "sm")
DRIVER_COLS: Tuple[str, ...] = tuple(f"hist_{d}" for d in DRIVER_NAMES)
FORECAST_WINDOW = 30
MAX_HISTORICAL_WINDOW = 365


class EpochDriverExtractor:
    """Extract per-sample driver weights and tag each with its calendar year."""

    # CSIF-slope thresholds (mirrors archive/generate_sensitivity_plots.py).
    # BET has weaker CSIF seasonality — use looser thresholds.
    _CSIF_MIN_DIFF: Dict[str, float] = {"BET": 0.05}
    _CSIF_MIN_SLOPE: Dict[str, float] = {"BET": 0.001}
    _CSIF_MIN_DIFF_DEFAULT: float = 0.20
    _CSIF_MIN_SLOPE_DEFAULT: float = 0.002

    def __init__(
        self,
        data_path: Path,
        pred_path: Path,
        coords_path: Path,
        sos_indices_path: Optional[Path],
        eos_indices_path: Optional[Path],
        output_dir: Path,
        pft: str,
        detect_csif: bool = False,
    ):
        self.data_path = Path(data_path)
        self.pred_path = Path(pred_path)
        self.coords_path = Path(coords_path)
        self.sos_indices_path = Path(sos_indices_path) if sos_indices_path else None
        self.eos_indices_path = Path(eos_indices_path) if eos_indices_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pft = pft
        self.detect_csif = detect_csif

        self.data: Optional[Dict] = None
        self.preds: Optional[Dict] = None
        self.coords: Optional[pd.DataFrame] = None
        self.sos_indices: Optional[np.ndarray] = None
        self.eos_indices: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        for label, path in [
            ("data", self.data_path),
            ("predictions", self.pred_path),
            ("coordinates", self.coords_path),
        ]:
            if not path.exists():
                logger.error(f"Missing {label} file: {path}")
                return False

        logger.info(f"Loading data:        {self.data_path}")
        with open(self.data_path, "rb") as fp:
            self.data = pickle.load(fp)

        logger.info(f"Loading predictions: {self.pred_path}")
        with open(self.pred_path, "rb") as fp:
            self.preds = pickle.load(fp)

        for key in ("attention_scores", "historical_selection_weights"):
            if key not in self.preds:
                logger.error(f"Predictions file missing required key: {key}")
                return False

        att_shape = self.preds["attention_scores"].shape
        hist_shape = self.preds["historical_selection_weights"].shape
        logger.info(f"  attention_scores              shape: {att_shape}")
        logger.info(f"  historical_selection_weights  shape: {hist_shape}")

        if hist_shape[2] != 7:
            logger.warning(
                f"Expected 7 historical features (CSIF + 6 drivers), got {hist_shape[2]}"
            )

        logger.info(f"Loading coordinates: {self.coords_path}")
        if self.coords_path.suffix == ".parquet":
            self.coords = pd.read_parquet(self.coords_path)
        else:
            self.coords = pd.read_csv(self.coords_path)
        self.coords = self.coords.drop_duplicates(subset=["location"], keep="first")
        logger.info(f"  {len(self.coords)} unique locations")

        if self.sos_indices_path and self.sos_indices_path.exists():
            self.sos_indices = np.load(self.sos_indices_path)
            logger.info(f"  Loaded {len(self.sos_indices)} SOS indices")
        if self.eos_indices_path and self.eos_indices_path.exists():
            self.eos_indices = np.load(self.eos_indices_path)
            logger.info(f"  Loaded {len(self.eos_indices)} EOS indices")

        if self.sos_indices is None and self.eos_indices is None:
            if not self.detect_csif:
                logger.error(
                    "Neither SOS nor EOS index files were provided. "
                    "Either supply --sos-indices / --eos-indices, "
                    "or pass --detect-csif to derive them from the CSIF time series."
                )
                return False
            logger.info(
                "  No index files provided; will detect SOS/EOS from CSIF slope."
            )

        return True

    # ------------------------------------------------------------------
    # Core: window logic + per-sample extraction
    # ------------------------------------------------------------------

    def _max_attention_window_start(self, index: int) -> int:
        """Slide a FORECAST_WINDOW-day window and return the highest-attention start."""
        att = np.mean(self.preds["attention_scores"][index], axis=0)
        max_start = min(
            MAX_HISTORICAL_WINDOW - FORECAST_WINDOW + 1,
            len(att) - FORECAST_WINDOW + 1,
        )
        # Vectorised: cumulative-sum trick gives all window sums in O(T).
        if max_start <= 0:
            return 0
        csum = np.concatenate(
            [[0.0], np.cumsum(att[: max_start + FORECAST_WINDOW - 1])]
        )
        window_sums = (
            csum[FORECAST_WINDOW : FORECAST_WINDOW + max_start] - csum[:max_start]
        )
        return int(np.argmax(window_sums))

    def _parse_year(self, raw_id) -> Optional[int]:
        """Pull a 4-digit year out of a test sample id of form `{location}_{ISO-date}`.

        Tolerant of edge cases: arrays, bytes, partial dates. Returns None on
        anything we can't parse; caller drops those samples and counts them.
        """
        if isinstance(raw_id, (list, tuple, np.ndarray)):
            raw_id = raw_id[0]
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("utf-8", errors="ignore")
        try:
            _, date_str = str(raw_id).split("_", 1)
        except ValueError:
            return None
        try:
            return pd.to_datetime(date_str).year
        except (ValueError, TypeError):
            # Fall back to the first 4 digits if Pandas can't parse it.
            digits = "".join(ch for ch in date_str if ch.isdigit())[:4]
            return int(digits) if len(digits) == 4 else None

    def _extract_phase(self, indices: np.ndarray, phase: str) -> pd.DataFrame:
        """Return one row per usable sample for the given phenological phase."""
        logger.info(f"Extracting {phase}: {len(indices)} candidate samples")

        att_array = self.preds["attention_scores"]
        hist_array = self.preds["historical_selection_weights"]
        ids = self.data["data_sets"]["test"]["id"]

        max_pred_index = len(att_array) - 1
        max_window_start = MAX_HISTORICAL_WINDOW - FORECAST_WINDOW

        records: List[Dict] = []
        skipped_bounds = skipped_loc = skipped_year = 0

        for idx_num, index in enumerate(indices):
            if index > max_pred_index:
                skipped_bounds += 1
                continue

            window_start = self._max_attention_window_start(int(index))
            if window_start > max_window_start:
                skipped_bounds += 1
                continue

            try:
                raw_id = ids[index][0] if hasattr(ids[index], "__len__") else ids[index]
                location_id = int(str(raw_id).split("_")[0])
            except (KeyError, IndexError, ValueError):
                skipped_loc += 1
                continue

            year = self._parse_year(raw_id)
            if year is None:
                skipped_year += 1
                continue

            row = {"location": location_id, "phase": phase, "year": year}
            window = hist_array[
                int(index), window_start : window_start + FORECAST_WINDOW, 1:7
            ]
            row.update(
                {
                    f"hist_{name}": float(np.median(window[:, i]))
                    for i, name in enumerate(DRIVER_NAMES)
                }
            )
            records.append(row)

            if (idx_num + 1) % 1000 == 0:
                logger.info(f"  {idx_num + 1}/{len(indices)} processed")

        logger.info(f"  Kept           : {len(records)}")
        logger.info(f"  Skipped bounds : {skipped_bounds}")
        logger.info(f"  Skipped loc id : {skipped_loc}")
        logger.info(f"  Skipped year   : {skipped_year}")

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df
        return df.merge(self.coords, on="location", how="left")

    # ------------------------------------------------------------------
    # CSIF-slope SOS/EOS detection (fallback when no .npy index files)
    # ------------------------------------------------------------------

    def _detect_from_csif(self) -> None:
        """Populate ``self.sos_indices`` and ``self.eos_indices`` by scanning
        the test CSIF values for rising/falling 30-sample windows.

        Replicates the method in ``archive/generate_sensitivity_plots.py``
        (``end_or_start_growing_season``). Works on the flattened test target
        array which is ordered as the TFT test-set rows.
        """
        logger.info("Detecting SOS/EOS indices from CSIF slope ...")
        csif = self.data["data_sets"]["test"]["target"].flatten().astype(np.float32)
        n = len(csif)
        fw = FORECAST_WINDOW

        # PFT-specific thresholds
        base_pft = self.pft.split("_")[0]
        min_diff = self._CSIF_MIN_DIFF.get(base_pft, self._CSIF_MIN_DIFF_DEFAULT)
        min_slope = self._CSIF_MIN_SLOPE.get(base_pft, self._CSIF_MIN_SLOPE_DEFAULT)

        sos_list: List[int] = []
        eos_list: List[int] = []

        # Pre-compute x for regression: shape (fw,)
        x = np.arange(fw, dtype=np.float32)
        x_mean = x.mean()
        x_denom = float(np.sum((x - x_mean) ** 2))

        for start in range(0, n - fw + 1, fw):
            batch = csif[start : start + fw]
            if abs(float(batch[-1]) - float(batch[0])) <= min_diff:
                continue
            y_mean = batch.mean()
            slope = float(np.sum((x - x_mean) * (batch - y_mean))) / x_denom
            idx = start // fw  # prediction-array index for this batch
            if slope >= min_slope:
                sos_list.append(idx)
            elif slope <= -(min_slope + 0.0005):
                eos_list.append(idx)

        self.sos_indices = np.array(sos_list, dtype=np.int64)
        self.eos_indices = np.array(eos_list, dtype=np.int64)
        logger.info(
            f"  Detected {len(self.sos_indices)} SOS and {len(self.eos_indices)} EOS events"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        if not self.load():
            return pd.DataFrame()

        if self.sos_indices is None and self.eos_indices is None:
            self._detect_from_csif()

        parts: List[pd.DataFrame] = []
        if self.sos_indices is not None and len(self.sos_indices):
            parts.append(self._extract_phase(self.sos_indices, "SOS"))
        if self.eos_indices is not None and len(self.eos_indices):
            parts.append(self._extract_phase(self.eos_indices, "EOS"))

        if not parts:
            logger.error("No samples extracted.")
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True)
        cols = ["location", "latitude", "longitude", "phase", "year", *DRIVER_COLS]
        return df[cols]

    def save(self, df: pd.DataFrame) -> Path:
        out = self.output_dir / f"{self.pft}_yearly_drivers.csv"
        df.to_csv(out, index=False)
        logger.info(f"Saved {out}  shape={df.shape}")
        return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract per-(location, phase, year) driver weights from TFT predictions."
    )
    p.add_argument(
        "--pft", required=True, help="Plant functional type identifier (e.g. BDT, BET)."
    )
    p.add_argument("--data-path", required=True, type=Path)
    p.add_argument("--pred-path", required=True, type=Path)
    p.add_argument("--coords-path", required=True, type=Path)
    p.add_argument("--sos-indices", type=Path)
    p.add_argument("--eos-indices", type=Path)
    p.add_argument(
        "--detect-csif",
        action="store_true",
        help="Derive SOS/EOS from CSIF slope rather than pre-computed index files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write {PFT}_yearly_drivers.csv",
    )

    args = p.parse_args()

    extractor = EpochDriverExtractor(
        data_path=args.data_path,
        pred_path=args.pred_path,
        coords_path=args.coords_path,
        sos_indices_path=args.sos_indices,
        eos_indices_path=args.eos_indices,
        output_dir=args.output_dir,
        pft=args.pft,
        detect_csif=args.detect_csif,
    )

    df = extractor.run()
    if df.empty:
        logger.error("Extraction produced no rows.")
        return 1

    extractor.save(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
