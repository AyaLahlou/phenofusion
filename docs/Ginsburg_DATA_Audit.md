# Ginsburg Data Audit

**Audit date:** 2026-06-01
**Auditor:** Claude Code (al4385)

---

## Audited Locations

1. [`/burg-archive/glab/users/al4385/`](#location-1-burg-archiveglabusers-al4385) — ~5.6 TB (Ginsburg scratch/glab storage)
2. [`/burg-archive/home/al4385/phenofusion/data/`](#location-2-burg-archivehomeal4385phenofusiondata) — ~5.3 GB (repo-local data outputs)

---

# Location 1: `/burg-archive/glab/users/al4385/`

**Total storage:** ~5.6 TB

---

## Abbreviations

| Code | Meaning |
|------|---------|
| BDT | Broadleaf Deciduous Trees |
| BET | Broadleaf Evergreen Trees |
| GRA | Grasslands |
| NDT | Needleleaf Deciduous Trees |
| NET | Needleleaf Evergreen Trees |
| SHR | Shrubs |
| SCH | Schrophytes (shrub-like PFT, used with SCH naming) |
| TFT | Temporal Fusion Transformer (model architecture) |
| PFT | Plant Functional Type |
| SOS | Start of Season |
| EOS | End of Season |
| CSIFMETEO | Combined SIF + meteorological input data (parquet format) |

Latitude band suffixes in filenames:
- `50_90` = 50°N–90°N (boreal)
- `50_20` = 20°N–50°N (temperate northern hemisphere)
- `-20_20` = 20°S–20°N (tropical)
- `-20_-60` = 20°S–60°S (southern mid-latitudes)

---

## Directory Tree Overview

```
/burg-archive/glab/users/al4385/
├── analysis/                   152 MB    Oct 2025
│   └── driversdata/
├── data/                       739 GB    Dec 2025
│   ├── 4GST/                   665 MB    Aug 2025
│   ├── CLM_ml_phenology/       (empty)   Jun 2025
│   ├── coordinates/            1.4 GB    Oct 2025
│   ├── CSIFMETEO/              120 GB    Nov 2025
│   ├── phenoformer/            346 MB    Jul 2025
│   ├── pyphenology_benchmark/  103 MB    Aug 2025
│   ├── pyphenology_formatted/  24 GB     Aug 2025
│   ├── sensitivity/            (empty)   Nov 2024
│   ├── soiltexture/            3.9 MB    Oct 2025
│   ├── sorted_TFT_30_40_BDTinterval/  (empty)  Mar 2025
│   ├── SOSEOS_change/          1.7 MB    Nov 2024
│   ├── TFT_30/                 90 GB     Oct 2025
│   ├── TFT_30_40_BDTinterval/  36 GB     Apr 2025
│   ├── TFT_30_40years/         87 GB     Jul 2025
│   ├── TFT_30_test/            12 GB     May 2024
│   ├── TFT_365/                45 GB     Apr 2024
│   └── TFT_40_overlapping_samples/  327 GB  Oct 2025
├── figures/                    58 MB     Nov 2025
│   ├── diagnostics/
│   └── drivermaps/
├── ms7073/                     65 GB     Jan 2026
│   ├── analysis/
│   ├── data/
│   ├── predictions/
│   └── weights/
├── phenology-ml-clm/           (empty)   Oct 2025
│   └── data/TFT_train_inputs_60_10/
├── predictions/                4.1 TB    Jan 2026
│   ├── pred_40year_moresamples/   3.4 TB
│   ├── predictions_40years/       231 GB
│   ├── pretrained_0331/           159 GB
│   ├── pretrained_1219/           69 GB
│   └── TFT_30_40years/            231 GB
├── test_pretraining.ipynb      Jan 2025
└── weights/                    384 MB    Oct 2025
    ├── pretrained_1219/        255 MB
    ├── TFT_30_0429/            87 MB
    └── TFT_30_1209/            43 MB
```

---

## Section 1: `data/`

### `data/CSIFMETEO/` — 120 GB | Last modified: Nov 2025 | **ACTIVE**

The primary input data directory. Contains merged CSI (CSIF solar-induced fluorescence proxy) + meteorological (METEO) time-series data in parquet format, organized by PFT. This is the upstream source for all TFT model inputs.

**Contents:**
| File/Dir | Size | Date | Notes |
|----------|------|------|-------|
| `sorted_merged_BDT_1982_2021.parquet` | 36.4 GB | Mar 2025 | Final sorted BDT merged input |
| `merged_BDT_1982_2021.parquet` | 21.9 GB | Dec 2024 | Unsorted version |
| `megred_BDT_no_location.parquet` | 21.9 GB | Dec 2024 | Typo in name; no location columns — legacy |
| `sorted_BDT_50_90_1982_2021.parquet` | 5.6 GB | Oct 2024 | Boreal BDT |
| `sorted_BDT_-20_20_merged_1982_2021.parquet` | 5.3 GB | Oct 2024 | Tropical BDT |
| `sorted_BDT_50_20_merged_1982_2021.parquet` | 5.4 GB | Oct 2024 | Temperate BDT |
| `sorted_BDT_-20_-60_1982_2021.parquet` | 2.5 GB | Oct 2024 | S. mid-lat BDT |
| `sorted_BDT_50_20_2000_2021_10loc.csv` | 8.4 MB | Nov 2025 | Small diagnostic subset |
| `GRA_merged.parquet` | 5.7 GB | Oct 2025 | Merged GRA input |
| `SCH_merged.parquet` | 1.6 GB | Oct 2025 | Merged SCH input |
| `sorted_GRA.parquet` | 7.2 GB | Oct 2025 | ms7073-created sorted GRA |
| `sorted_SHR.parquet` | 2.1 GB | Oct 2025 | ms7073-created sorted SHR |
| `BET_merged/` | ~4.2 GB | Oct 2024 | Merged BET input (`sorted_BET_merged_1982_2021.parquet`) |
| `NDT_merged/` | ~1.3 GB | Oct 2024 | Merged NDT input (`sorted_NDT_merged_1982_2021.parquet`) |
| `NET_merged/` | ~2.8 GB | Oct 2024 | Merged NET input (`sorted_NET_merged_1982_2021.parquet`) |
| `GRA/` | ~3.5 GB | Oct 2025 | Per-year parquet files (`full_YYYY_daily_NoNAN.parquet`, 1982–2021) |
| `SCH/` | ~1.0 GB | Oct 2025 | Per-year parquet files (same format as GRA) |

**Created by:** `src/phenofusion/utils/parquet_merger.py`, `src/phenofusion/dataio/TFT_process_40years.py`, preprocessing notebooks

**Used by:**
- `bashscripts/sos_eos/find_sos_eos_BDT.sh` → reads `sorted_merged_BDT_1982_2021.parquet`
- `bashscripts/sos_eos/find_sos_eos_NDT.sh` → reads `NDT_merged/sorted_NDT_merged_1982_2021.parquet`
- `bashscripts/sos_eos/find_sos_eos_NET.sh` → reads `NET_merged/sorted_NET_merged_1982_2021.parquet`
- `bashscripts/sos_eos/find_sos_eos_GRA.sh` → reads `GRA_merged.parquet`
- `bashscripts/processing/process_BDT_2020.sh` → reads `sorted_BDT_-20_20_merged_1982_2021.parquet`
- `bashscripts/processing/process_BDT_2060.sh` → reads `sorted_BDT_-20_-60_1982_2021.parquet`
- `bashscripts/processing/process_GRA.sh` → reads `sorted_GRA.parquet`
- `bashscripts/diagnostics/diagnose_southern_midlat.sh` → reads `sorted_merged_BDT_1982_2021.parquet`
- `debugging/create_subset.py` → reads `sorted_BDT_50_90_1982_2021.parquet` and `sorted_BDT_50_20_merged_1982_2021.parquet`
- `src/phenofusion/utils/get_coordinates.py` → reads parquets to extract coordinates
- `src/phenofusion/dataio/TFT_process_40years.py` → reads CSIFMETEO, writes TFT_30_40years

**Potential cleanup:**
- `megred_BDT_no_location.parquet` (typo, appears superseded by `merged_BDT_1982_2021.parquet`)
- The Oct 2024 per-latband parquet files are likely superseded by `sorted_merged_BDT_1982_2021.parquet` (Mar 2025)

---

### `data/coordinates/` — 1.4 GB | Last modified: Oct 2025 | **ACTIVE**

Per-PFT coordinate lookup files (lat/lon per location index). Used by all driver map and analysis scripts to map model outputs back to geographic positions.

**Contents:**
| File | Size | Date | Notes |
|------|------|------|-------|
| `SHR.parquet` | 598 KB | Oct 2025 | ms7073-created |
| `Sch.parquet` | 598 KB | Oct 2025 | al4385-created (same as SHR?) |
| `GRA.parquet` | 8.5 MB | Oct 2025 | |
| `merged_BDT.parquet` | 1.3 GB | Jan 2025 | Large; used by analysis scripts |
| `BDT_temperate.parquet` | 3.0 MB | May 2024 | Early version |
| `NET.parquet` | 8.9 MB | May 2014 | |
| `BDT_-20_-60.parquet` | 4.0 MB | May 2024 | |
| `BDT_50_90.parquet` | 23.8 MB | May 2024 | |
| `BDT_50_20.parquet` | 19.6 MB | May 2024 | |
| `BET.parquet` | 16.1 MB | May 2024 | |
| `BDT_-20_20.parquet` | 19.8 MB | May 2024 | |
| `NDT.parquet` | 4.0 MB | May 2024 | |

**Created by:** `src/phenofusion/utils/get_coordinates.py`

**Used by:** All driver map bash scripts (`bashscripts/drivers/`), `bashscripts/analysis/sensitivity.sh`, `bashscripts/analysis/create_analysisdf.sh`, `src/phenofusion/analysis/run_driver_maps.py`, `src/phenofusion/analysis/generate_sensitivity_plots.py`, `src/phenofusion/analysis/diagnose_latitude_bands.py`

---

### `data/TFT_40_overlapping_samples/` — 327 GB | Last modified: Oct 2025 | **ACTIVE (current standard)**

The canonical TFT model input dataset with 40-year overlapping sample windows. This is the primary training/inference dataset used in the most recent (ms7073 / pretrained_1219) model runs. Contains one `.pkl` per PFT.

**Contents:**
| File | Size | Date | Owner | Notes |
|------|------|------|-------|-------|
| `GRA_1982_2021.pkl` | 31.4 GB | Oct 2025 | ms7073 | |
| `SHR_1982_2021.pkl` | 9.3 GB | Oct 2025 | ms7073 | |
| `BDT_50_20_1982_2021.pkl` | 61.3 GB | Aug 2025 | al4385 | |
| `BDT_50_90_1982_2021.pkl` | 65.4 GB | Aug 2025 | al4385 | |
| `BDT_-20_-60_1982_2021.pkl` | 21.6 GB | Aug 2025 | al4385 | |
| `BET_1982_2021.pkl` | 47.7 GB | Aug 2025 | al4385 | |
| `NET_1982_2021.pkl` | 33.8 GB | Aug 2025 | al4385 | |
| `NDT_1982_2021.pkl` | 15.4 GB | Aug 2025 | al4385 | |
| `BDT_-20_20_1982_2021.pkl` | 64.0 GB | Aug 2025 | al4385 | |

**Created by:** `bashscripts/processing/process_BDT_2020.sh`, `process_BDT_2060.sh`, `process_BET.sh`, `process_GRA.sh`, `process_NET.sh`, `process_NDT.sh`, `process_SHR.sh`

**Used by:** All `bashscripts/drivers/drivers_*.sh`, `bashscripts/prediction/predict_*.sh`, `bashscripts/analysis/sensitivity.sh`, `debugging/create_analysis_dataframe.py`, notebooks `driversdata_diagnostics.ipynb`, `get_analysis_df.ipynb`

---

### `data/TFT_30/` — 90 GB | Last modified: Oct 2025 | **PARTIALLY ACTIVE**

Earlier-era TFT input data (30-year lookback window format). Mostly superseded by TFT_40_overlapping_samples, but the `.pkl` files (not pickle) for GRA and SHR were created here in Oct 2025 by ms7073, and BDT pickle files date to Dec 2024.

**Contents (key files):**
| File | Size | Date | Notes |
|------|------|------|-------|
| `sorted_GRA.pkl` | 5.1 GB | Oct 2025 | ms7073-created |
| `sorted_SHR.pkl` | 1.5 GB | Oct 2025 | ms7073-created |
| `SHR.pkl` | 35 KB | Oct 2025 | Stub/index only? |
| `GRA.pkl` | 35 KB | Oct 2025 | Stub/index only? |
| `merged_BDT_1982_2021.pkl` | 34.6 GB | Dec 2024 | |
| `BDT_*.pickle` (×4 lat bands) | ~7.7 GB each | Apr 2024 | Older format |
| `BET.pickle` | 7.7 GB | Apr 2024 | Older format |
| `NET.pickle` | 5.5 GB | Apr 2024 | Older format |
| `NDT.pickle` | 2.5 GB | Apr 2024 | Older format |
| `TFT_30_test/` | subdir | Oct 2024 | Test subset |

**Created by:** Data processing scripts (earlier pipeline), `src/phenofusion/dataio/TFT_process_40years.py`

**Used by:** `bashscripts/training/train_gra.sh`, `train_SHR.sh`, `train_shr.sh` (all read from `TFT_30/`), `bashscripts/metrics/metrics_GRA.sh`, `metrics_SHR.sh`, `bashscripts/analysis/analysis.sh`

**Status:** The Apr 2024 `.pickle` files and the `TFT_30_test/` subdir are effectively obsolete; the Dec 2024 BDT pkl and Oct 2025 GRA/SHR sorted pkls are still referenced by training scripts.

---

### `data/TFT_30_40years/` — 87 GB | Last modified: Jul 2025 | **PARTIALLY ACTIVE**

Intermediate dataset: 40-year span data formatted for the TFT-30 model window. BDT has been superseded by `TFT_40_overlapping_samples`, but the merged BDT pkl is still referenced by several analysis scripts.

**Contents:**
| File | Size | Date | Notes |
|------|------|------|-------|
| `sorted_merged_BDT_1982_2021.pkl` | 37.7 GB | Jul 2025 | Still referenced by analysis |
| `NDT.pickle` | 2.7 GB | Oct 2024 | Older NDT |
| 6 more PFT `.pickle` files | ~50 GB total | Oct 2024 | BET, NET, BDT bands, SHR |

**Created by:** `src/phenofusion/dataio/TFT_process_40years.py`

**Used by:** `bashscripts/prediction/predict_BDT_30_40years.sh`, `bashscripts/analysis/create_analysisdf.sh`, `bashscripts/drivers/drivermap_BET.sh`, `drivermap_NDT.sh`, `drivermap_NET.sh`, `debugging/save_sos_eos_indices_fixed.py`, `debugging/compare_old_new_indices.py`, `src/phenofusion/analysis/generate_sensitivity_plots.py`, `sensitivity_plot.py`, `run_sensitivity_analysis.py`

---

### `data/TFT_30_40_BDTinterval/` — 36 GB | Last modified: Apr 2025 | **OBSOLETE**

BDT-only interval-band dataset, an intermediate step between TFT_30 and TFT_40_overlapping_samples.

**Contents:** 4 `.pkl` files covering BDT lat bands (50_90, -20_20, 50_20, and a combined), Apr 2025.

**Created by:** Earlier processing scripts (`process_BDT_5020.sh`, `process_BDT_5090.sh`)

**Used by:** `bashscripts/prediction/predict_5020.sh` (first block, now superseded), `predict_5090.sh` (first block), `predict_2020.sh` (first block), `predict_march31st.sh`

**Status:** Superseded by `TFT_40_overlapping_samples/`. The prediction scripts that reference it all have a second (newer) block pointing to `TFT_40_overlapping_samples`. **Safe to archive or delete.**

---

### `data/TFT_30_test/` — 12 GB | Last modified: May 2024 | **OBSOLETE**

Early test dataset with a subset of PFTs (NDT, BDT lat bands). Used during initial model development in 2024.

**Contents:** 7 `.pickle` files (BDT_-20_20, BDT_50_20, BDT_50_90, BET, NET, NDT), May 2024

**Used by:** `src/phenofusion/analysis/drivers_map_dec.py` (hardcoded path, but this appears to be a legacy/commented analysis script)

**Status:** Fully superseded. **Safe to delete.**

---

### `data/TFT_365/` — 45 GB | Last modified: Apr 2024 | **OBSOLETE**

Very early dataset using 365-day lookback window format. All files from Apr 2024.

**Contents:** 5 `.parquet` files for BDT (2 lat bands), BET, NET, NDT

**Created by:** Earlier preprocessing

**Used by:** Nothing in current scripts (no grep hits for `TFT_365` in active scripts)

**Status:** Fully superseded by TFT_40_overlapping_samples. **Safe to delete.** This is 45 GB of recoverable space.

---

### `data/sorted_TFT_30_40_BDTinterval/` — **EMPTY** | Last modified: Mar 2025 | **OBSOLETE**

Empty directory. Likely a staging directory that was never populated or was cleared.

**Status:** **Safe to delete.**

---

### `data/4GST/` — 665 MB | Last modified: Aug 2025 | **RESEARCH / BENCHMARK**

Predictions from the 4GST model (a competing/benchmark phenology model). Used for comparison against TFT predictions.

**Contents:** `NET_predictions.pkl`, `BDT_predictions.pkl`, `BET_predictions.pkl`, `NDT_predictions.pkl` (all Aug 2025)

**Created by:** Likely generated externally or via a separate 4GST model script

**Used by:** Benchmark comparison notebooks/scripts

**Status:** Benchmark reference data. Keep unless the comparison analysis is complete.

---

### `data/phenoformer/` — 346 MB | Last modified: Jul 2025 | **RESEARCH / BENCHMARK**

PhenoFormer model predictions and weights for benchmark comparison. Contains predictions for BDT, BET, NET, NDT plus model weights (`.pth`) and metric outputs.

**Contents:** 4 PFT prediction `.pkl` files, 4 `.pth` weight files, 2 metric files (PNG + TXT)

**Created by:** PhenoFormer model training/inference (separate model benchmark)

**Used by:** Benchmark comparison analysis

**Status:** Benchmark reference. Keep if PhenoFormer comparison is still ongoing.

---

### `data/pyphenology_formatted/` — 24 GB | Last modified: Aug 2025 | **RESEARCH / BENCHMARK**

Data formatted for the pyphenology baseline model. Contains train/validation/test splits for BDT, BET, NDT, NET (12 pkl files total).

**Contents:** `{PFT}_{train|validation|test}_pyphenology.pkl` for BDT, BET, NDT, NET

**Created by:** Data reformatting scripts for pyphenology benchmark

**Used by:** pyphenology benchmark training/evaluation

**Status:** Benchmark reference. Keep if pyphenology comparison is ongoing.

---

### `data/pyphenology_benchmark/` — 103 MB | Last modified: Aug 2025 | **RESEARCH / BENCHMARK**

Output of the pyphenology benchmark (ThermalTime model). Contains `BET_ThermalTime_predictions.pkl` and a `metrics/` subdir with comparison figure.

**Created by:** pyphenology benchmark run

**Used by:** Comparison analysis

**Status:** Keep for manuscript figures/benchmarking.

---

### `data/soiltexture/` — 3.9 MB | Last modified: Oct 2025 | **ACTIVE**

Processed GLDAS soil texture data at 0.25° resolution. Used as a supplementary covariate in model inputs or analysis.

**Contents:** `GLDASp5_soiltexture_025d_processed.parquet`

**Created by:** Soil texture preprocessing script (not found in phenofusion bashscripts; likely a one-off Jupyter notebook)

**Used by:** Likely model input processing or spatial analysis

---

### `data/SOSEOS_change/` — 1.7 MB | Last modified: Nov 2024 | **OUTDATED**

Early SOS/EOS phenology change detection outputs from Nov 2024. Small pkl files with phenological change dates for various PFTs.

**Contents:** 7 `.pkl` files for BDT, NDT, NET, BET (all Nov 2024)

**Created by:** Early SOS/EOS detection scripts

**Used by:** `debugging/save_sos_eos_indices.py` (reads from `ms7073/analysis/driversdata/undersampling/` not this dir, so this dir may be orphaned)

**Status:** Superseded by the Jul–Aug 2025 SOS/EOS CSVs in `analysis/driversdata/`. **Potentially safe to delete.**

---

### `data/CLM_ml_phenology/` — **EMPTY** | Last modified: Jun 2025 | **OBSOLETE**

Empty directory. Likely placeholder for CLM model phenology comparison that was never populated.

**Status:** **Safe to delete.**

---

### `data/sensitivity/` — **EMPTY** | Last modified: Nov 2024 | **OBSOLETE**

Empty directory. Sensitivity analysis outputs go to `ms7073/analysis/sensitivity_analysis_oversampling/` instead.

**Status:** **Safe to delete.**

---

## Section 2: `weights/`

### `weights/pretrained_1219/` — 255 MB | Last modified: Oct 2025 | **ACTIVE (current)**

The canonical model weights directory. Contains the most recent trained weights for all PFTs. The name "1219" refers to Dec 19 (training date). The BDT checkpoint (`weights_merged_BDT_1982_2021_feb2025_checkpoint.pth`) is used as the transfer learning starting point for all other PFTs.

**Contents:**
| File | Size | Date | Notes |
|------|------|------|-------|
| `weights_sorted_GRA_3.pth` | 19.6 MB | Oct 2025 | Latest GRA weights |
| `weights_sorted_SHR_3.pth` | 19.6 MB | Oct 2025 | Latest SHR weights |
| `weights_BET_3.pth` | 19.6 MB | Feb 2025 | Latest BET weights |
| `weights_NDT_3.pth` | 19.5 MB | Feb 2025 | Latest NDT weights |
| `weights_NET_3.pth` | 19.6 MB | Feb 2025 | Latest NET weights |
| `weights_merged_BDT_1982_2021_feb2025_checkpoint.pth` | 19.6 MB | Feb 2025 | Main BDT pretrained checkpoint |
| `weights_BET_2.pth`, `weights_NET_2.pth`, `weights_NDT_2.pth` | ~19.5 MB each | Jan 2025 | Previous iteration |
| `weights_BET.pth`, `weights_NET.pth`, `weights_NDT.pth` | ~19.5 MB each | Jan 2025 | First fine-tuned versions |
| `weights_merged_BDT_1982_2021_3.pth` | 19.6 MB | Jan 2025 | BDT v3 checkpoint |
| `weights_merged_BDT_1982_2021.pth` | 13.0 MB | Dec 2024 | BDT initial weights (smaller architecture?) |

**Used by:** All `bashscripts/prediction/predict_*.sh`, `bashscripts/training/train_*.sh`

---

### `weights/TFT_30_0429/` — 87 MB | Last modified: May 2024 | **OBSOLETE**

Weights from April 2024 TFT-30 training run. All files dated Apr–May 2024.

**Contents:** 7 `.pth` files for BDT (lat bands), BET, NET

**Status:** Superseded by pretrained_1219. **Safe to archive or delete.**

---

### `weights/TFT_30_1209/` — 43 MB | Last modified: Dec 2024 | **OBSOLETE**

December 2024 intermediate weights for NDT and BET only. Superseded by pretrained_1219 versions.

**Status:** Superseded. **Safe to delete.**

---

## Section 3: `predictions/`

### `predictions/pred_40year_moresamples/` — 3.4 TB | Last modified: Jan 2026 | **ACTIVE (current, but bulky)**

The primary predictions directory for the current pipeline. Contains TFT model predictions for all PFTs over the 1982–2021 period. **This is the largest directory (3.4 TB).**

Root-level full prediction files:
| File | Size | Date | Notes |
|------|------|------|-------|
| `BDT_50_20_1982_2021_attention_only.pkl` | 224 GB | Jan 2026 | Attention scores only (extracted by `extract_attention_scores.py`) |
| `BDT_50_90_1982_2021.pkl` | 296 GB | Nov 2025 | Full BDT boreal predictions |
| `BDT_-20_20_1982_2021.pkl` | 289 GB | Oct 2025 | Full BDT tropical predictions |
| `BDT_50_20_1982_2021.pkl` | 277 GB | Oct 2025 | Full BDT temperate predictions |
| `GRA_1982_2021.pkl` | 142 GB | Oct 2025 | Full GRA predictions |
| `SHR_1982_2021.pkl` | 24 GB | Oct 2025 | Full SHR predictions |
| `BET_1982_2021.pkl` | 216 GB | Oct 2025 | Full BET predictions |
| `NET_1982_2021.pkl` | 153 GB | Oct 2025 | Full NET predictions |
| `GRA_20251024.pkl` | 142 GB | Oct 2025 | ms7073 run — duplicate of GRA_1982_2021.pkl? |
| `SHR_20251023.pkl` | 42 GB | Oct 2025 | ms7073 run — older SHR version |
| `BDT_-20_-60_1982_2021.pkl` | 99 GB | Oct 2025 | Full BDT S. mid-lat predictions |
| `NDT_1982_2021.pkl` | 70 GB | Oct 2025 | Full NDT predictions (no `_merged` suffix) |

Per-PFT checkpoint subdirs (`BDT/`, `BET/`, `GRA/`, `NET/`, `NDT/`, `SHR/`): contain batch-level checkpoint prediction files used during long inference runs. These are intermediate outputs and can be cleaned up once final `.pkl` files are confirmed good.

**Created by:** `bashscripts/prediction/predict_5020.sh`, `predict_5090.sh`, `predict_2020.sh`, `predict_2060.sh`, `predict_BET.sh`, `predict_NET.sh`, `predict_NDT.sh`, `predict_GRA.sh`, `predict_SHR.sh`; `scripts/extract_attention_scores.py`

**Used by:** All `bashscripts/drivers/drivers_*.sh`, `bashscripts/analysis/sensitivity.sh`, `bashscripts/metrics/agg_pred_check*.sh`, `notebooks/driversdata_diagnostics.ipynb`, `notebooks/get_analysis_df.ipynb`

**Potential cleanup:**
- `GRA_20251024.pkl` (142 GB, ms7073 run) appears to duplicate `GRA_1982_2021.pkl`
- `SHR_20251023.pkl` (42 GB) predates `SHR_1982_2021.pkl` (24 GB) — older run, likely safe to delete
- All checkpoint batch files in `BDT/`, `BET/`, `GRA/`, `NET/`, `SHR/` subdirs can be deleted once final PKLs are confirmed good

---

### `predictions/predictions_40years/` — 231 GB | Last modified: Jan 2026 | **ACTIVE**

Predictions from an earlier pipeline iteration (TFT-30 model on 40-year data). Contains BDT, BET, NDT only.

| Subdir | File | Size | Date | Notes |
|--------|------|------|------|-------|
| `BDT/` | `sorted_merged_BDT_1982_2021_TFT30_40years.pkl` | 170 GB | Jan 2026 | Very large |
| `BET/` | `preds.pickle` | 38 GB | Jun 2025 | |
| `NDT/` | `preds.pickle` | 12 GB | Jun 2025 | |

**Created by:** `bashscripts/prediction/predict_BDT_30_40years.sh`, `predict_5020.sh` (second block), `drivermap_BET.sh`, `drivermap_NDT.sh`, `drivermap_NET.sh`

**Used by:** `bashscripts/analysis/create_analysisdf.sh`, `drivermap_BET.sh`, `drivermap_NDT.sh`, `drivermap_NET.sh`

---

### `predictions/pretrained_0331/` — 159 GB | Last modified: Apr 2025 | **OBSOLETE**

Predictions from the March 31 2025 model run. Contains 3 BDT lat band predictions.

**Contents:** `sorted_BDT_50_90_1982_2021.pkl` (52 GB), `pred_BDT_-20_20_merged...pkl` (51 GB), `pred_BDT_50_20_merged...pkl` (49 GB)

**Created by:** `bashscripts/prediction/predict_5090.sh` (first block), `predict_5020.sh` (first block), `predict_march31st.sh`, `predict_2020.sh` (first block)

**Status:** Superseded by `pred_40year_moresamples/`. **Safe to delete** — this is 159 GB of recoverable space.

---

### `predictions/pretrained_1219/` — 69 GB | Last modified: Oct 2025 | **PARTIALLY ACTIVE**

Predictions from the pretrained_1219 model, before the `pred_40year_moresamples` directory was set up. Contains GRA, SHR, and older BET/NDT/NET predictions.

| File | Size | Date | Notes |
|------|------|------|-------|
| `GRA_20251013.pkl` | 5.8 GB | Oct 2025 | Used by `metrics_GRA.sh` |
| `SHR_20251009.pkl` | 1.7 GB | Oct 2025 | Used by `metrics_SHR.sh` |
| `pred_NET_2_jan232025.pkl` | 6.2 GB | Feb 2025 | Older NET |
| Others (NDT, BET, etc.) | ~55 GB | Jan–Feb 2025 | Older runs |

**Used by:** `bashscripts/metrics/metrics_GRA.sh` (reads `GRA_20251013.pkl`), `metrics_SHR.sh` (reads `SHR_20251009.pkl`), `bashscripts/analysis/analysis.sh`

**Status:** GRA and SHR files still referenced by metrics scripts. Older files (pre-Oct 2025) likely superseded.

---

### `predictions/TFT_30_40years/` — 231 GB | Last modified: Oct 2024 | **OBSOLETE**

Very early predictions (Oct 2024) from first TFT-30 model on 40-year data. Contains BET, NET, NDT, BDT.

**Contents:** 7 pkl files, all dated Oct 2024

**Used by:** `src/phenofusion/analysis/sensitivity_plot.py`, `generate_sensitivity_plots.py`, `run_sensitivity_analysis.py` (all have hardcoded defaults pointing here, but these scripts have newer `--pred_dir` arguments)

**Status:** Superseded. However, 231 GB — verify sensitivity analysis scripts don't still rely on these before deleting. The source scripts allow overriding via `--pred_dir`.

---

## Section 4: `analysis/`

### `analysis/driversdata/` — 152 MB | Last modified: Oct 2025 | **ACTIVE (intermediate analysis)**

Driver analysis outputs for al4385's own runs (as opposed to the ms7073 collaborative analysis in `ms7073/analysis/`). Contains SOS/EOS CSV files for BDT (all lat bands), BET, NDT, NET, and driver map PNGs — all from Jul–Aug 2025.

**Contents:** ~55 SOS/EOS CSV files (dated 20250714–20250722) plus PNG figures, and oversampling analysis CSVs from Dec 2025

**Created by:** `bashscripts/sos_eos/find_sos_eos_*.sh` scripts

**Used by:** Driver map generation scripts, analysis notebooks

**Status:** Jul–Aug 2025 CSVs appear to be intermediate/exploratory versions; Dec 2025 files (`BDT_50_20_analysisdf.csv`, `BDT_50_90_analysisdf.csv`) are current outputs of `create_analysisdf.sh`.

---

## Section 5: `ms7073/`

Shared collaborative directory used by user `ms7073` (another group member). Contains their own copies of data, weights, predictions, and driver analysis outputs.

### `ms7073/data/` — ~12 GB | Last modified: Oct 2025

Mirror of key data files, created by ms7073 for their own runs:
- `CSIFMETEO/`: sorted GRA and SHR parquets + dummy subset
- `TFT_30/`: sorted GRA pkl
- `TFT_40_overlapping_samples/`: GRA/SHR parquets and dummy subset

### `ms7073/weights/pretrained_1219/` — 19 MB | Last modified: Oct 2025

One weight file: `weights_sorted_SHR_dev.pth` (dev version of SHR weights)

### `ms7073/predictions/pretrained_1219/` — 1.6 GB | Last modified: Oct 2025

`SHR_20251009.pkl` — ms7073's SHR predictions (also mirrored to `predictions/pretrained_1219/`)

### `ms7073/analysis/driversdata/` — 65 GB | Last modified: Dec 2025 | **ACTIVE**

The main joint driver analysis outputs. Organized by sampling strategy and date:

| Subdir | Files | Date | Notes |
|--------|-------|------|-------|
| `oversampling/` | 66 files | Oct–Dec 2025 | Main analysis: SOS/EOS CSVs for all PFTs + analysis dataframes |
| `undersampling/` | 10 files | Oct–Nov 2025 | Undersampling sensitivity check |
| `driver_maps_oversampling/` | 10 files | Oct 2025 | Initial driver maps |
| `driver_maps_oversampling_20251107/` | 14 files | Nov 2025 | Updated Nov 7 |
| `driver_maps_oversampling_20251203/` | 14 files | Dec 2025 | Updated Dec 3 |
| `driver_maps_oversampling_20251204/` | 14 files | Dec 2025 | Most recent dated run |
| `dec12025/` | 4 files | Dec 2025 | Dec 1 outputs |
| `driver_maps_oversampling_check_20251029/` | 10 files | Oct 2025 | Oct 29 check run |
| `driver_maps_oversampling_latlon/` | 10 files | Oct 2025 | Lat/lon variant |
| `driver_maps_oversampling_slope_no_BET_specific/` | 10 files | Oct 2025 | BET-excluded slope analysis |
| `driver_maps_undersampling_check_20251030/` | 4 files | Oct 2025 | Undersampling check |
| `moresamples/` | 0 (empty) | Oct 2025 | Placeholder |

**Created by:** `bashscripts/drivers/drivers_*.sh`, `notebooks/clean/sos_eos_detection_refactored.py`, `debugging/save_sos_eos_indices.py`

**Used by:** `notebooks/driversdata_diagnostics.ipynb`, `notebooks/get_analysis_df.ipynb`, driver map generation scripts

### `ms7073/analysis/sensitivity_analysis_oversampling/` — 4.5 MB | Last modified: Oct 2025

6 sensitivity analysis PNG figures (temperature, water availability, solar radiation for SOS and EOS), created Oct 2025 by ms7073.

---

## Section 6: `figures/`

### `figures/drivermaps/` — ~58 MB | Last modified: Dec 2025 | **ACTIVE (figures output)**

Rendered driver map figures organized by run date:

| Subdir | Date | Contents |
|--------|------|---------|
| `dec12025/` | Dec 2025 | Most recent driver maps for all PFTs (created by `drivers_BDT_2020.sh`, `drivers_BDT_5020.sh`) |
| `nov242025_2/` | Nov 2025 | Nov 24 second pass |
| `nov242025/` | Nov 2025 | Nov 24 first pass |
| `GRA_test_11212025/` | Nov 2025 | GRA-specific test run |
| `test_11062025/` | Nov 2025 | BET test run (via `drivers_BET_new.sh`) |
| `test_attmapv2/` | Oct 2025 | Attention map v2 test |
| `test/` | Oct 2025 | General test outputs |
| Root-level PNGs | Oct 2025 | SOS/EOS maps for all PFTs (All_PFTs, BDT_Combined, GRA, SHR, NDT, NET, BET) |

**Created by:** `src/phenofusion/analysis/run_driver_maps.py` (called by driver bash scripts), `notebooks/driversdata_diagnostics.ipynb`

### `figures/diagnostics/` — ~300 KB | Last modified: Dec 2025 | **ACTIVE**

2 diagnostic PNGs comparing BDT_50_20 vs BDT_50_90 SOS/EOS distributions.

**Created by:** `bashscripts/diagnostics/diagnose_latitude_bands.sh`, `src/phenofusion/analysis/diagnose_latitude_bands.py`

---

## Section 7: `phenology-ml-clm/`

### `phenology-ml-clm/` — **EMPTY** | Last modified: Oct 2025

Empty directory structure (has `data/TFT_train_inputs_60_10/` subdir but no files). Likely a placeholder for a CLM-ML phenology comparison project.

**Status:** **Safe to delete.**

---

## Section 8: Root-level file

### `test_pretraining.ipynb` — 377 bytes | Last modified: Jan 2025

A very small (377-byte) notebook, almost certainly empty or minimal. Used during early pretraining experiments.

**Status:** Likely safe to delete.

---

## Section 9: `data/sorted_BDT_50_20_2000_2021_10loc.csv` (root of `data/`)

| File | Size | Date | Notes |
|------|------|------|-------|
| `sorted_BDT_50_20_2000_2021_10loc.csv` | 366 bytes (in data root) | Dec 2025 | 10-location diagnostic subset |

A small CSV index file (not in a subdir) at the root of `data/`. Likely a diagnostic subset created during investigation of BDT 50_20 predictions. Not referenced by any script.

---

## Summary: Storage by Category

| Category | Approx. Size | Status |
|----------|-------------|--------|
| `predictions/pred_40year_moresamples/` | 3.4 TB | Active but bulky; checkpoint subdirs can be cleaned |
| `predictions/predictions_40years/` | 231 GB | Partially active |
| `predictions/TFT_30_40years/` | 231 GB | Obsolete |
| `predictions/pretrained_0331/` | 159 GB | **Deletable** |
| `predictions/pretrained_1219/` | 69 GB | Partially active |
| `data/TFT_40_overlapping_samples/` | 327 GB | Active — canonical input data |
| `data/CSIFMETEO/` | 120 GB | Active — raw merged inputs |
| `data/TFT_30_40years/` | 87 GB | Partially active |
| `data/TFT_30/` | 90 GB | Partially active |
| `data/TFT_365/` | 45 GB | **Deletable** |
| `data/TFT_30_40_BDTinterval/` | 36 GB | **Deletable** |
| `data/pyphenology_formatted/` | 24 GB | Benchmark data |
| `data/TFT_30_test/` | 12 GB | **Deletable** |
| `ms7073/` | 65 GB | Active (collaborative) |
| `data/4GST/` | 665 MB | Benchmark data |
| `data/phenoformer/` | 346 MB | Benchmark data |
| `weights/pretrained_1219/` | 255 MB | Active — current weights |
| `weights/TFT_30_0429/` | 87 MB | **Deletable** |
| `analysis/driversdata/` | 152 MB | Active |
| `figures/` | 58 MB | Active |
| `weights/TFT_30_1209/` | 43 MB | **Deletable** |
| Everything else | ~10 MB | See notes above |

---

## Recommended Cleanup (Priority Order)

1. **High-value, clearly safe:**
   - `predictions/pretrained_0331/` — 159 GB, superseded
   - `data/TFT_365/` — 45 GB, superseded (no script references)
   - `data/TFT_30_40_BDTinterval/` — 36 GB, superseded
   - `data/TFT_30_test/` — 12 GB, superseded
   - `weights/TFT_30_0429/` and `weights/TFT_30_1209/` — 130 MB
   - Empty directories: `data/CLM_ml_phenology/`, `data/sensitivity/`, `data/sorted_TFT_30_40_BDTinterval/`, `phenology-ml-clm/`

2. **Verify before deleting:**
   - `predictions/TFT_30_40years/` — 231 GB; sensitivity scripts have hardcoded defaults to it
   - Checkpoint batch files in `predictions/pred_40year_moresamples/BDT/`, `BET/`, `GRA/`, `NET/`, `SHR/`
   - Duplicate predictions: `GRA_20251024.pkl` (142 GB) and `SHR_20251023.pkl` (42 GB) in `pred_40year_moresamples/`

3. **Low priority / keep:**
   - All benchmark data (`4GST/`, `phenoformer/`, `pyphenology_formatted/`, `pyphenology_benchmark/`)
   - `data/SOSEOS_change/` — small (1.7 MB), may still be useful as reference

**Estimated recoverable space:** ~800 GB immediately (pretrained_0331 + TFT_365 + TFT_30_40_BDTinterval + TFT_30_test + old weights), up to ~1.4 TB if checkpoint files and duplicate predictions are also removed.

---

---

# Location 2: `/burg-archive/home/al4385/phenofusion/data/`

**Total storage:** ~5.3 GB
**Purpose:** Repo-local output data — processed analysis inputs/outputs that are small enough to live inside the phenofusion project directory rather than in glab scratch storage. These are derived outputs (not raw model predictions or training data).

```
phenofusion/data/
├── analysis_dataframes/        5.2 GB    Jan 2026
│   └── BDT_analysisdf.csv
├── drivers_data/               43 MB     Jan 2026
│   ├── BDT_drivers_EOS.csv
│   ├── BDT_drivers_SOS.csv
│   ├── BDT_drivers_2_EOS.csv
│   ├── BDT_drivers_2_SOS.csv
│   ├── BDT_EOS_driver_correlations.png
│   ├── BDT_EOS_driver_distributions.png
│   ├── BDT_EOS_spatial_coverage.png
│   ├── BDT_SOS_driver_correlations.png
│   ├── BDT_SOS_driver_distributions.png
│   └── BDT_SOS_spatial_coverage.png
├── SOS_EOS_indices/            28 KB     Jan 2026
│   ├── SOS_pred_indices_BET.npy
│   ├── EOS_pred_indices_BET.npy
│   ├── NDT/
│   │   ├── SOS_pred_indices_NDT.npy
│   │   └── EOS_pred_indices_NDT.npy
│   └── NET/                    (empty)
└── README.md                   (empty)
```

---

## `data/analysis_dataframes/` — 5.2 GB | Last modified: Jan 28 2026 | **ACTIVE**

Contains the merged analysis DataFrame for BDT, which joins model predictions with input features and coordinates for downstream driver analysis.

**Contents:**
| File | Size | Date | Notes |
|------|------|------|-------|
| `BDT_analysisdf.csv` | 5.2 GB | Jan 28 2026 | Merged analysis table: predictions + features + coords for all BDT lat bands |

**Created by:** `bashscripts/analysis/create_analysisdf.sh`, which calls `src/phenofusion/analysis/create_analysis_dataframe.py` with:
- Input data: `glab/.../data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl`
- Input predictions: `glab/.../predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl`
- Input coords: `glab/.../data/coordinates/merged_BDT.parquet`

**Used by:**
- `debugging/save_sos_eos_indices.py` — reads this CSV to extract SOS/EOS window indices, then writes to `SOS_EOS_indices/`
- `debugging/save_sos_eos_indices_fixed.py` — fixed version of the same script
- `debugging/diagnose_sos_eos_detection_failure.py` — reads for diagnostic analysis of index detection failures

**Notes:** Only BDT is present. The `create_analysisdf.sh` script targets BDT explicitly; equivalent files for other PFTs would need separate runs. At 5.2 GB this is large for a repo-local file — it is derived from glab data and could be regenerated.

---

## `data/drivers_data/` — 43 MB | Last modified: Jan 29 2026 | **ACTIVE**

Climate driver importance CSVs and diagnostic figures for BDT. These are the extracted per-location, per-year driver attribution tables used for the final driver maps.

**Contents:**
| File | Size | Date | Notes |
|------|------|------|-------|
| `BDT_drivers_EOS.csv` | 14 MB | Jan 29 2026 | First extraction run (EOS phase) |
| `BDT_drivers_SOS.csv` | 14 MB | Jan 29 2026 | First extraction run (SOS phase) |
| `BDT_drivers_2_EOS.csv` | 6.8 MB | Jan 29 2026 | Second run (EOS) — smaller, likely filtered/refined |
| `BDT_drivers_2_SOS.csv` | 6.5 MB | Jan 29 2026 | Second run (SOS) — smaller, likely filtered/refined |
| `BDT_EOS_driver_correlations.png` | 62 KB | Jan 29 2026 | Correlation heatmap (EOS) |
| `BDT_EOS_driver_distributions.png` | 188 KB | Jan 29 2026 | Distribution plots (EOS) |
| `BDT_EOS_spatial_coverage.png` | 895 KB | Jan 29 2026 | Spatial coverage map (EOS) |
| `BDT_SOS_driver_correlations.png` | 61 KB | Jan 29 2026 | Correlation heatmap (SOS) |
| `BDT_SOS_driver_distributions.png` | 192 KB | Jan 29 2026 | Distribution plots (SOS) |
| `BDT_SOS_spatial_coverage.png` | 853 KB | Jan 29 2026 | Spatial coverage map (SOS) |

**Created by:** `bashscripts/drivers/extract_drivers_refactored.sh`, which calls `debugging/driver_data_modified.py` (and `scripts/get_driver_map_data_refactored.py`). The script reads:
- Data: `glab/.../data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl`
- Predictions: `glab/.../predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl`
- Coordinates: `glab/.../data/coordinates/merged_BDT.parquet`
- SOS/EOS indices: from `debugging_data_artifacts/debugging_subset_data/`

**Used by:** Downstream driver map notebooks and analysis (`notebooks/clean/get_driver_map.ipynb`, `notebooks/diagnostics/driversdata_sos_eos_from_msc.ipynb`)

**Notes:**
- The `_2_` variants are a second extraction run — the smaller file size suggests a subset or filtered output. The original `_drivers_EOS/SOS.csv` files may be superseded by the `_2_` versions.
- `drivers_BDT_json.sh` writes to `/burg/glab/users/al4385/phenofusion/data/drivers_data/` (a different path using `/burg/glab/` prefix), so its output does NOT land here.
- Only BDT is present; other PFTs have not been run through this pipeline yet.

---

## `data/SOS_EOS_indices/` — 28 KB | Last modified: Jan 30 2026 | **ACTIVE**

Small numpy index arrays mapping detected SOS/EOS events to indices in the prediction arrays. These are the lookup tables used by the driver extraction pipeline to know which prediction timesteps correspond to phenological transitions.

**Contents:**
| File | Size | Date | Notes |
|------|------|------|-------|
| `SOS_pred_indices_BET.npy` | 128 bytes | Jan 30 2026 | BET SOS indices |
| `EOS_pred_indices_BET.npy` | 128 bytes | Jan 30 2026 | BET EOS indices |
| `NDT/SOS_pred_indices_NDT.npy` | 128 bytes | Jan 30 2026 | NDT SOS indices |
| `NDT/EOS_pred_indices_NDT.npy` | 128 bytes | Jan 30 2026 | NDT EOS indices |
| `NET/` | (empty) | Jan 30 2026 | NET directory exists but has no files yet |

**Created by:** `debugging/save_sos_eos_indices.py` — reads `analysis_dataframes/BDT_analysisdf.csv` (for BDT) and `ms7073/analysis/driversdata/undersampling/` CSV files (for NDT, NET, BET), then saves numpy arrays of prediction indices.

**Used by:** `bashscripts/drivers/extract_drivers_refactored.sh` → `debugging/driver_data_modified.py` (reads `SOS_pred_indices` and `EOS_pred_indices` npy files to select prediction windows for driver extraction)

**Notes:**
- The 128-byte file size for all `.npy` files is suspicious — each file stores only a NumPy array header plus a small number of int32 values (128 bytes = 96 bytes header + ~8 indices). This may indicate these are placeholder/test outputs or that the index arrays are very short. Worth verifying the index counts are correct before using for production driver extraction.
- The `NET/` directory is empty — NET indices have not been computed yet.
- BDT indices live in `debugging_data_artifacts/debugging_subset_data/` (not here), so this directory currently only has BET and NDT.

---

## `data/README.md` — Empty | **PLACEHOLDER**

Empty file. No documentation has been written here yet.

---

## Summary: `phenofusion/data/` Storage

| Path | Size | Status |
|------|------|--------|
| `analysis_dataframes/BDT_analysisdf.csv` | 5.2 GB | Active; derived from glab data, regenerable |
| `drivers_data/BDT_drivers_*.csv` (×4) | ~41 MB | Active; `_2_` versions likely supersede originals |
| `drivers_data/*.png` (×6) | ~2 MB | Active diagnostic figures |
| `SOS_EOS_indices/**/*.npy` (×4) | ~512 bytes | Active; verify index counts before relying on them |

**Key gaps:** Only BDT is fully processed through `analysis_dataframes/` and `drivers_data/`; BET, NDT, NET, GRA, SHR are missing from both. `SOS_EOS_indices/NET/` is empty.
