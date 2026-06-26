# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Phenofusion is a deep-learning phenology forecasting toolkit. It trains a **Temporal Fusion Transformer (TFT)** on CSIF (Clear-Sky Solar-Induced Fluorescence) + meteorological time series to predict vegetation phenology — Start of Season (SOS) and End of Season (EOS) — for multiple **Plant Functional Types (PFTs)**: BDT, BET, NET, NDT, GRA, SHR.

The workflow is: raw parquet data → TFT processing (sliding windows, scaling) → `.pkl` → model training → predictions `.pkl` → SOS/EOS detection → driver/attention analysis → driver maps & heatmaps.

Large data files, model weights, and predictions are stored externally at `/burg/glab/users/al4385/` (not in this repo). Jobs run on Columbia's **Burg SLURM cluster**.

---

## Commands

### Linting & Formatting
```bash
ruff check .          # lint
black --check .       # format check
black .               # apply formatting
mypy src/phenofusion  # type check
```

### Tests
```bash
pytest tests/                                  # run all tests
pytest tests/test_heatmap_analysis.py -v       # single test file
pytest tests/test_metrics_tft_refactored.py -v
```

### Install (development)
```bash
pip install -e ".[dev]"
```

### Data Processing (run via SLURM)
```bash
sbatch bashscripts/processing/process_BDT_2020.sh  # or any other PFT/band
# Underlying Python:
python src/phenofusion/dataio/TFT_process.py \
  --data_path /path/to/sorted_PFT.parquet \
  --output_path /path/to/output/ \
  --hist_len 365 --fut_len 30 --samp_interval 395
```

### Training (run via SLURM)
```bash
sbatch bashscripts/training/train.sh  # uses train_tft_TL.py for transfer learning
# Underlying Python (transfer learning from pretrained BDT weights):
python src/phenofusion/models/train_tft_TL.py <PFT>.pickle --checkpoint <checkpoint.pth>
# Training from scratch:
python src/phenofusion/models/train_tft.py <PFT>.pickle
```

### Prediction (run via SLURM)
```bash
sbatch bashscripts/prediction/predict_BET.sh
# Underlying Python:
python src/phenofusion/dataio/predict_tft.py \
  --training_dir /path/to/training_data.pkl \
  --data_dir /path/to/test_data.pkl \
  --weights_dir /path/to/weights.pth \
  --pred_dir /path/to/output.pkl \
  [--start_from_checkpoint]
```

### Driver/Attention Analysis (SLURM)
```bash
sbatch bashscripts/drivers/drivers_BET.sh
# Underlying Python:
python src/phenofusion/dataio/driversdata.py \
  --pred_path /path/to/pred.pkl \
  --data_path /path/to/data.pkl \
  --coord_path /path/to/coords.parquet \
  --output_path /path/to/output_prefix \
  --forecast_window_length 30 \
  --PFT BET
```

### Attention Window Extraction (SLURM)
Extracts the 30-day window of maximum attention per sample and saves `window_start` indices to CSV.
Requires pre-computed SOS/EOS index `.npy` files (from `bashscripts/drivers/generate_sos_eos_indices_*.sh`).
```bash
sbatch bashscripts/drivers/get_attention_window_BDT.sh  # or BET, NET, NDT, GRA, SHR
# Underlying Python:
python scripts/get_attention_window_data.py \
  --data-path /path/to/data.pkl \
  --pred-path /path/to/pred.pkl \
  --coords-path /path/to/coords.parquet \
  --sos-indices /path/to/SOS_pred_indices.npy \
  --eos-indices /path/to/EOS_pred_indices.npy \
  --output-dir data/attention_window/ \
  --output-prefix BDT \
  --pft BDT
# Outputs: data/attention_window/BDT_SOS.csv, BDT_EOS.csv
#          + BDT_SOS_window_distribution.png, BDT_SOS_window_spatial.png (diagnostics)
```

### Sensitivity Shift Analysis (two-stage)

**Stage 1 — epoch driver extraction** (run on HPC where predictions live):
```bash
python scripts/extract_epoch_drivers.py \
  --pft BDT \
  --data-path /path/to/data.pkl \
  --pred-path /path/to/pred.pkl \
  --coords-path /path/to/coords.parquet \
  --sos-indices data/SOS_EOS_indices/SOS_pred_indices_BDT.npy \
  --eos-indices data/SOS_EOS_indices/EOS_pred_indices_BDT.npy \
  --output-dir /path/to/sensitivity_data/
# Or use --detect-csif to derive SOS/EOS from CSIF slope instead of .npy files
# Output: {PFT}_yearly_drivers.csv  (one row per sample, with year column)
```
rsync the resulting CSVs back into the repo (`Data/data_output/sensitivity_data/`).

**Stage 2 — figure generation** (runs locally on the rsynced CSVs):
```bash
python scripts/run_sensitivity_shifts.py \
  --data-dir Data/data_output/sensitivity_data/ \
  --output-dir Figures/figures_output/sensitivity/ \
  --early-start 1982 --early-end 1991 \
  --late-start 2012 --late-end 2021 \
  --bootstrap-B 200 \
  --projection Robinson

# Skip specific tiers or aggregates:
python scripts/run_sensitivity_shifts.py --skip-tier 3 mosaic
# Restrict to specific PFTs:
python scripts/run_sensitivity_shifts.py --pfts BDT BET
```

> `run_sensitivity_shifts.py` imports `from phenofusion.sensitivity_shifts import ...` with `src/` on sys.path. The canonical module is `src/phenofusion/sensitivity_shifts.py`.

### Attention Window Maps (SLURM)
Generates global scalar colormaps showing *when* (which 30-day window) the model paid maximum attention.
Reads the CSVs produced by the extraction step above.
```bash
sbatch bashscripts/drivers/generate_window_maps_BDT.sh
# Underlying Python:
python scripts/generate_window_maps.py \
  --data-dir data/attention_window/ \
  --output-dir data/attention_window/maps/ \
  --projection Robinson \
  --cmap plasma \
  --dpi 300
# Outputs per-PFT PNGs, combined BDT PNG, all-PFTs PNG,
#         and a 2×3 multi-panel figure with shared colorbar
```

### Analysis & Visualization
```bash
# Driver maps (RGB: temperature=R, solar=G, precipitation=B)
python src/phenofusion/analysis/run_driver_maps.py \
  --data-dir /path/to/attention_csvs/ \
  --output-dir ./plots/

# Sensitivity analysis (compare two time periods)
python src/phenofusion/analysis/run_sensitivity_analysis.py \
  --year1 1985 --year2 2020 --seasons sos eos

# Heatmap analysis
python src/phenofusion/analysis/run_heatmap_analysis.py \
  --city "Budapest" --pft "BDT" --season "sos"

# Metrics plots
python src/phenofusion/analysis/metrics_tft.py \
  --filename pred_BDT.pkl \
  --data_dir /path/to/data.pkl \
  --pred_dir /path/to/pred.pkl \
  --fig_dir /path/to/figure.png
```

### Monitor SLURM Jobs
```bash
squeue -u al4385
sacct -u al4385 --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f *.out   # live output
tail -f *.err   # live errors
```

---

## Architecture & Data Flow

### TFT Data Format
`TFT_process.py` converts raw parquet files into a nested dict saved as `.pkl`:
```python
{
  "data_sets": {
    "train": {"historical_ts_numeric": ..., "future_ts_numeric": ..., "target": ..., "id": ..., ...},
    "validation": {...},
    "test": {...}
  },
  "feature_map": {...},   # maps feature categories to column names
  "scalers": {...},       # fitted sklearn scalers per column
  "categorical_cardinalities": {...}
}
```
**Feature categories:**
- `static_feats_numeric`: latitude, longitude
- `static_feats_categorical`: soil type
- `historical_ts_numeric`: tmin, tmax, radiation, precipitation, swvl1 (soil moisture), photoperiod, sif_clear_inst
- `future_ts_numeric`: tmin, tmax, radiation, precipitation, swvl1, photoperiod (known future meteorology)
- `target`: sif_clear_inst over the forecast window

**Data splits (default):** train = <1992 + ≥2011; validation = 1992–2001; test = 2001–2011.
**40-year test mode** (`--test_mode`): test = 1982–2022, no train/validation.

### Model Configuration
TFT model (`tft_torch.tft.TemporalFusionTransformer`) defaults:
- `hist_len=365`, `fut_len=30`, `state_size=160`, `lstm_layers=4`, `attention_heads=4`
- Output quantiles: [0.1, 0.5, 0.9]
- Loss: quantile loss via `tft_torch.loss.get_quantiles_loss_and_q_risk`
- Optimizer: AdamW, warmup+cosine LR schedule
- Training uses WandB for logging

Predictions `.pkl` contain:
- `predicted_quantiles`: shape `(N, fut_len, 3)` — three quantile predictions
- `attention_scores`: shape `(N, hist_len+fut_len, ...)` — used for driver analysis
- `historical_selection_weights`: shape `(N, hist_len, n_features)` — per-feature importance

### SOS/EOS Detection and Driver Analysis
`driversdata.py` (`AnalysisWorkflow.run_analysis`) orchestrates:
1. Loads data + predictions
2. Detects SOS/EOS events per PFT:
   - **Generic PFTs**: slope-based detection (sign of slope within 30-day batches)
   - **BET**: latitude-DOY band matching (`PhenologyConfig.BET_PHENOLOGY_BANDS`)
3. For each detected event, extracts the 30-day window of maximum attention (`AttentionAnalyzer.max_attention_window`)
4. Records per-driver median selection weights for that window → CSV output

### Sensitivity Shift Pipeline
Answers "**how has phenology's sensitivity to each driver changed over time?**" Compares two fixed epochs (default 1982–1991 vs 2012–2021) across all PFTs and phases. Two scripts, both in `scripts/`:

**Stage 1 — `extract_epoch_drivers.py`** (`EpochDriverExtractor`):
- Key difference from `get_driver_map_data_refactored.py`: it parses the ISO date in each sample ID (`{location}_{ISO-date}`) to attach a **calendar year** to every sample, then emits **one row per sample** rather than a per-location median.
- Output schema: `location, latitude, longitude, phase, year, hist_tmin, hist_tmax, hist_rad, hist_precip, hist_photo, hist_sm`
- Output file: `{PFT}_yearly_drivers.csv`
- Two SOS/EOS detection modes:
  - Pre-computed `.npy` index files (`--sos-indices` / `--eos-indices`) — preferred when they exist
  - CSIF-slope detection (`--detect-csif`) — slides a 30-sample window over the flattened test CSIF target; flags rising slope as SOS, falling as EOS using PFT-specific thresholds (looser for BET)
- The yearly CSV is small enough to rsync back from HPC to `Data/data_output/sensitivity_data/` for local plotting.

**Stage 2 — `sensitivity_shifts.py`** (`SensitivityShiftAnalyzer` + `SensitivityShiftPlotter`):
- Reads `{PFT}_yearly_drivers.csv` files; driver weights are row-normalized into three composites: `temp` (tmin+tmax), `sol` (rad+photo), `p` (precip+sm).
- `SensitivityShiftAnalyzer` computes:
  - Per-pixel **epoch medians** (inner-joined so only pixels with data in both epochs contribute to Δ)
  - **Δ = late − early** per driver group
  - **Bootstrapped 95% CIs** (B=200, chunked over pixels) — marks `d{g}_sig=True` where CI excludes 0
  - **Dominance codes** — argmax of {temp, sol, p} per epoch, 3×3 transition codes (0–8), and margin (top − second weight)
  - **Rolling decadal trajectories** — zonal-mean composites over sliding decade windows
- `SensitivityShiftPlotter` generates three tiers:
  - **Tier 1**: 3-panel per-driver Δ maps with stippling (sig pixels); categorical 3×3 dominance-flip map; all-PFT mosaic
  - **Tier 2**: latitude-binned zonal Δ profiles with IQR; |Δ| magnitude maps; PFT×phase×driver summary heatmap
  - **Tier 3**: dominance margin Δ map; rolling decadal trajectory heatmap (lat × time × driver weight)

**`run_sensitivity_shifts.py`** — top-level runner that auto-discovers CSVs, instantiates analyzer + plotter, loops over all (PFT, phase) combinations, then writes the all-PFT mosaic and summary heatmap. Default input: `Data/data_output/sensitivity_data/`. Default output: `Figures/figures_output/sensitivity/`.

**Import path issue**: `run_sensitivity_shifts.py` imports `from phenofusion.sensitivity_shifts import ...` with `src/` on sys.path, so `sensitivity_shifts.py` needs to be at `src/phenofusion/sensitivity_shifts.py` — it is currently at `scripts/sensitivity_shifts.py`.

### Attention Window Pipeline
Distinct from the driver maps pipeline. Answers "**when** does the model pay attention?" rather than "which driver matters most?".

Two steps, both in `scripts/` (not `src/phenofusion/`):

1. **`get_attention_window_data.py`** (`AttentionWindowExtractor`):
   - Requires pre-computed SOS/EOS index `.npy` files (prediction-level indices, one per sample).
   - For each sample index, averages `attention_scores` across forecast heads, then slides a 30-day window over the 365-day historical axis to find the position of maximum summed attention.
   - Deduplicates per location (median of `window_start`), merges with coordinate parquet.
   - Outputs: `{PFT}_SOS.csv` and `{PFT}_EOS.csv` (columns: `location`, `latitude`, `longitude`, `window_start`) stored in `data/attention_window/`.
   - Also writes diagnostic plots: histogram of `window_start` distribution and a spatial scatter colored by `window_start`.

2. **`generate_window_maps.py`** (`WindowMapGenerator`):
   - Reads the CSVs from step 1, builds a full 720×1440 global grid (0.25° resolution), and renders scalar colormaps with the `plasma` colormap.
   - Applies the same PFT latitude filters as driver maps.
   - Generates: per-PFT maps, combined BDT map (merging four latitude bands), all-PFTs combined map, and a 2×3 multi-panel figure with a shared colorbar.
   - Outputs saved to `data/attention_window/maps/`.

`window_start` is a 0-based index into the 365-day historical window; `window_start=0` means attention peaks at the very start of the historical record, `~335` means just before the forecast horizon.

### Driver Maps
`generate_driver_maps.py` reads the per-location driver CSVs and generates RGB images:
- **Red**: temperature importance (tmin + tmax)
- **Green**: solar importance (radiation + photoperiod)
- **Blue**: precipitation importance (precipitation + soil moisture)

Latitude filtering is applied per PFT (e.g., BET ≤30°, NET ≤75°).

### Spatial Data Conventions
- Location IDs are integer indices; `coordinates/*.parquet` files map them to lat/lon.
- BDT is split into latitude bands (e.g., `50_20`, `20_20`, `5020`, `5090`) because the global dataset is too large to process as one unit.
- Input parquets are named like `sorted_BET_merged_1982_2021.parquet`.

### Transfer Learning
`train_tft_TL.py` loads a pretrained BDT checkpoint then fine-tunes on another PFT. The pretrained checkpoint lives at `/burg/glab/users/al4385/weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth`.

### Subprojects
`subprojects/` contains self-contained experiments (e.g., `bdt_indiana_subset`) that have their own `bashscripts/`, `data/`, `outputs/`, and `scripts/` and are excluded from git tracking. Use `subprojects/create_subproject.sh` to scaffold new ones.

### Analysis Module (`src/phenofusion/analysis/`)
- `metrics_tft.py` — evaluates predictions: overall, across-site, seasonal, anomaly R²/RMSE/bias
- `generate_driver_maps.py` + `run_driver_maps.py` — RGB driver maps
- `generate_sensitivity_plots.py` + `run_sensitivity_analysis.py` — temporal sensitivity differences between years
- `heatmap_analysis.py` + `run_heatmap_analysis.py` — feature importance + attention heatmaps per city/region
- `config.py` — shared `PlotConfig`, `DataConfig`, `MetricsConfig` dataclasses

---

## Data Directory Reference

All paths are under `/burg-archive/glab/users/al4385/` (Ginsburg scratch; ~5.6 TB total) unless marked *in-repo*. Full audit: `docs/Ginsburg_DATA_Audit.md`.

### Filename conventions

Latitude band suffixes used across all file names:
- `50_90` = 50°N–90°N (boreal)
- `50_20` = 20°N–50°N (temperate north)
- `-20_20` = 20°S–20°N (tropical)
- `-20_-60` = 20°S–60°S (southern mid-latitudes)

BDT is always split across these four bands because the global dataset is too large to process as one unit. Other PFTs are single files.

### Active data directories

| Path | Size | Contents |
|------|------|----------|
| `data/CSIFMETEO/` | 120 GB | **Primary inputs.** Merged CSIF + meteorological parquets per PFT. Key files: `sorted_merged_BDT_1982_2021.parquet` (36 GB), `sorted_BDT_{band}_*.parquet`, `sorted_BET_merged_1982_2021.parquet`, `sorted_GRA.parquet`, `sorted_SHR.parquet`, etc. |
| `data/TFT_40_overlapping_samples/` | 327 GB | **Canonical TFT input pickles (current standard).** One `.pkl` per PFT/band: `BDT_50_20_1982_2021.pkl` (61 GB), `BET_1982_2021.pkl` (48 GB), `GRA_1982_2021.pkl` (31 GB), etc. Used by all prediction and driver scripts. |
| `data/TFT_30_40years/` | 87 GB | **Partially active.** Contains `sorted_merged_BDT_1982_2021.pkl` (38 GB) still referenced by several analysis/create-analysisdf scripts. Oct 2024 `.pickle` files for BET/NET/NDT are obsolete. |
| `data/TFT_30/` | 90 GB | **Partially active.** `sorted_GRA.pkl` and `sorted_SHR.pkl` (ms7073, Oct 2025) still used by GRA/SHR training and metrics scripts. Apr 2024 `.pickle` files are obsolete. |
| `data/coordinates/` | 1.4 GB | Per-PFT coordinate lookup parquets (location ID → lat/lon). Key file: `merged_BDT.parquet` (1.3 GB). Used by all driver and analysis scripts. Created by `src/phenofusion/utils/get_coordinates.py`. |
| `weights/pretrained_1219/` | 255 MB | **Current model weights.** `weights_merged_BDT_1982_2021_feb2025_checkpoint.pth` is the BDT pretrained base used for transfer learning. Per-PFT weights: `weights_BET_3.pth`, `weights_NET_3.pth`, `weights_NDT_3.pth`, `weights_sorted_GRA_3.pth`, `weights_sorted_SHR_3.pth`. |
| `predictions/pred_40year_moresamples/` | 3.4 TB | **Primary prediction outputs (current).** Full predictions for all PFTs: `BDT_50_20_1982_2021.pkl` (277 GB), `BET_1982_2021.pkl` (216 GB), `GRA_1982_2021.pkl` (142 GB), etc. Also contains `BDT_50_20_1982_2021_attention_only.pkl` (224 GB, attention scores extracted by `scripts/extract_attention_scores.py`). Per-PFT checkpoint subdirs (`BDT/`, `BET/`, etc.) hold batch intermediates. |
| `predictions/predictions_40years/` | 231 GB | **Partially active.** Contains `BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl` (170 GB) still referenced by `create_analysisdf.sh` and `drivermap_*.sh`. BET and NDT `preds.pickle` files also active. |
| `analysis/driversdata/` | 152 MB | al4385's driver analysis outputs. Dec 2025 files (`BDT_50_20_analysisdf.csv`, `BDT_50_90_analysisdf.csv`) are current. |
| `ms7073/analysis/driversdata/oversampling/` | ~65 GB | **Main joint driver analysis outputs.** SOS/EOS CSVs for all PFTs + analysis dataframes; most recent dated runs in `driver_maps_oversampling_20251204/`. Used by driver map generation. |
| `figures/drivermaps/` | 58 MB | Rendered driver map PNGs by run date. Most recent: `dec12025/`. |

### In-repo data directories

| Path | Contents |
|------|----------|
| `data/attention_window/` | `{PFT}_SOS.csv` / `{PFT}_EOS.csv` with window_start indices + coordinates. Diagnostic PNGs. Maps in `data/attention_window/maps/`. |
| `data/SOS_EOS_indices/` | `.npy` arrays of prediction-level SOS/EOS indices per PFT (`SOS_pred_indices_{PFT}.npy`, `EOS_pred_indices_{PFT}.npy`). Currently populated for BET and NDT; NET dir is empty. Note: 128-byte files indicate very small index arrays — verify counts before using in production. |
| `data/analysis_dataframes/` | `BDT_analysisdf.csv` (5.2 GB) — merged predictions + features + coordinates for BDT. Input to `debugging/save_sos_eos_indices.py`. |
| `data/drivers_data/` | `BDT_drivers_{SOS,EOS}.csv` and `BDT_drivers_2_{SOS,EOS}.csv` — per-location driver attribution tables for BDT (the `_2_` variants are the refined second run). Diagnostic PNGs. |

### Obsolete directories (safe to delete)

| Path | Size | Reason |
|------|------|--------|
| `predictions/pretrained_0331/` | 159 GB | Superseded by `pred_40year_moresamples/` |
| `data/TFT_365/` | 45 GB | No script references; fully superseded |
| `data/TFT_30_40_BDTinterval/` | 36 GB | Superseded by `TFT_40_overlapping_samples/` |
| `data/TFT_30_test/` | 12 GB | Early dev subset, superseded |
| `weights/TFT_30_0429/`, `weights/TFT_30_1209/` | 130 MB | Superseded by `pretrained_1219/` |

### Benchmark / comparison data (keep)

| Path | Contents |
|------|----------|
| `data/4GST/` | 4GST model predictions for BDT, BET, NET, NDT |
| `data/phenoformer/` | PhenoFormer predictions, weights, metrics |
| `data/pyphenology_formatted/` + `data/pyphenology_benchmark/` | ThermalTime baseline model data |
