# Phenofusion AI Coding Agent Instructions

## Project Overview
Phenofusion is a phenology forecasting framework using Temporal Fusion Transformer (TFT) models to predict vegetation phenology (Start/End of Season) from climate drivers. Runs on Columbia's SLURM HPC cluster.

## Pipeline Architecture (4 stages)
```
TFT_process.py → train_tft*.py → predict_tft.py → driversdata.py + visualizations
```

| Stage | Key File | Output Location |
|-------|----------|-----------------|
| Process | `src/phenofusion/dataio/TFT_process.py` | `/burg/glab/users/al4385/data/TFT_40_overlapping_samples/` |
| Train | `src/phenofusion/models/train_tft.py` or `train_tft_TL.py` | `/burg/glab/users/al4385/weights/` |
| Predict | `src/phenofusion/dataio/predict_tft.py` | `/burg/glab/users/al4385/predictions/` |
| Analyze | `src/phenofusion/dataio/driversdata.py` | Analysis CSVs with attention-derived driver importance |

## Critical Conventions (MUST FOLLOW)

### Feature Order (never reorder)
```python
["CSIF", "tmin", "tmax", "solar_radiation", "precipitation", "photoperiod", "soil_moisture"]
```
This ordering is hardcoded in model weights and attention analysis. Changing order breaks predictions.

### Data Processing Standards
- **Soil moisture**: Input files may have `sm` or `swvl1`; `TFT_process.py` normalizes to `swvl1`
- **Time windows**: Historical = 365 days, forecast = 30 days (see `hist_len`, `fut_len` params)
- **Sampling interval**: Default `samp_interval=15` for sliding windows in `TFT_process.py`

### Prediction Pickle Keys
- `historical_selection_weights`: shape (samples, 365, 7) - encoder attention
- `future_selection_weights`: shape (samples, 30, 7) - decoder attention  
- `attention_scores`: cross-attention weights for driver importance
- `predicted_quantiles`: shape (samples, forecast_len, 3) - [0.1, 0.5, 0.9] quantiles

### PFT-Specific Configurations

#### Thresholds in `driversdata.py`
```python
# PhenologyConfig class - BET/SHR have weaker signals
PFT_THRESHOLDS = {
    "BET": {"min_diff": 0.05, "min_slope": 0.001},
    "SHR": {"min_diff": 0.05, "min_slope": 0.001},
}
# Others use DEFAULT_MIN_DIFF=0.20, DEFAULT_MIN_SLOPE=0.002
```

#### BDT Latitude Bands
BDT splits into 4 latitude subgroups for analysis:
- `BDT_50_90`: 50°N to 90°N (high northern latitudes)
- `BDT_50_20`: 20°N to 50°N (mid-latitudes)
- `BDT_-20_20`: -20° to 20° (tropics)
- `BDT_-20_-60`: -60° to -20°S (southern hemisphere)

#### BET Phenology Bands
BET uses latitude-specific DOY ranges defined in `PhenologyConfig.BET_PHENOLOGY_BANDS` (23 bands from 90°N to -90°S). Example:
```python
(30.5, 35): {"sos": (40, 140), "eos": (210, 310)}  # DOY ranges for 30.5-35°N
```

## SLURM Workflow

### Module Loading (order matters!)
```bash
module load anaconda           # load first
module load cuda11.8/toolkit   # cuda11.1 for older scripts
pip install tft-torch          # always reinstall in job (not in shared env)
pip install cartopy            # only for visualization jobs
```

### Standard SBATCH Headers
```bash
#SBATCH --account=glab
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-15:00         # adjust based on PFT size
#SBATCH -C mem768              # high-memory nodes for data processing

# GPU training only:
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100s     # or leave unset for any GPU
```

### Complete PFT Workflow
```bash
# Step 1: Process raw data into TFT format
sbatch bashscripts/process_BET.sh

# Step 2: Train model (or skip if using pretrained weights)
sbatch bashscripts/train_gra.sh  # example for GRA with transfer learning

# Step 3: Generate predictions
sbatch bashscripts/predict_BET.sh

# Step 4: Extract driver importance and generate maps
sbatch bashscripts/drivers_BET.sh
```

### Train vs Test Mode
- **Training mode** (default): Uses 30-year train/validation split
- **Test mode** (`--test_mode` flag): Uses full 40-year data for evaluation
  - Pass `--test_mode` to `TFT_process.py` and `driversdata.py` for consistency

## Development Workflow

### Testing
```bash
pytest tests/ -v                              # full suite
pytest tests/test_heatmap_analysis.py -v      # heatmap/attention tests
pytest tests/test_metrics_tft_refactored.py   # metrics calculation tests
```
Tests use synthetic data; see `tests/test_heatmap_analysis.py` for fixture patterns (pytest fixtures in `conftest.py` are currently empty).

### Code Organization
```
src/phenofusion/
├── analysis/       # Post-processing: driver maps, sensitivity, metrics
├── dataio/         # Data pipeline: process, predict, driversdata
├── models/         # Training: train_tft.py, train_tft_TL.py
├── config/         # Configuration classes
├── utils/          # Shared utilities
└── cli.py          # Command-line interface (Typer-based)
```

### Adding New Analysis
Create in `subprojects/` (auto-gitignored) or use `subprojects/example_subproject/` as template:
```bash
cd subprojects
mkdir my_analysis && cd my_analysis
# Add notebooks, scripts, data - all ignored by git
```
See `docs/SUBPROJECTS_GUIDE.md` for full workflow.

## Key Files for Common Changes

| Change | File(s) |
|--------|---------|
| SOS/EOS detection thresholds | `src/phenofusion/dataio/driversdata.py` (see `PhenologyConfig`) |
| Transfer learning schedule | `src/phenofusion/models/train_tft_TL.py` (`setup_gradual_unfreezing`) |
| BET latitude bands | `driversdata.py` → `PhenologyConfig.BET_PHENOLOGY_BANDS` |
| Soil moisture normalization | `TFT_process.py` line 60-72 (sm → swvl1 mapping) |
| Driver map RGB colors | `src/phenofusion/analysis/generate_driver_maps.py` (temp=red, solar=green, precip=blue) |
| Model hyperparameters | `train_tft*.py` or config files in `src/phenofusion/config/` |
| WandB logging | `train_tft*.py` and `train_tft_TL.py` - logs metrics to WandB dashboard |

## Architecture Details

### Data Flow
1. **Raw data** (parquet): CSIF + 6 climate drivers per location/time
2. **TFT_process.py**: Creates sliding windows → pickled dict with train/val/test splits
3. **train_tft*.py**: Trains TFT model, saves checkpoints to `/burg/glab/.../weights/`
4. **predict_tft.py**: Loads model, generates predictions + attention weights
5. **driversdata.py**: Analyzes attention → CSVs with driver importance per location
6. **generate_driver_maps.py**: Creates RGB visualizations from CSVs

### Transfer Learning
`train_tft_TL.py` implements gradual unfreezing:
- Start: All layers frozen except quantile outputs
- Progressively unfreeze: LSTM decoders → variable selection → encoders
- Different learning rates: `lr_pretrained=1e-4`, `lr_new=1e-3`
- Example: Train BDT from scratch → transfer to GRA/SHR with less data

### Attention Mechanism
TFT uses 3 attention types:
1. **Variable selection**: `historical_selection_weights`, `future_selection_weights`
2. **Cross-attention**: `attention_scores` (temporal attention across historical window)
3. Driver importance = avg(variable_selection_weights) over max-attention 30-day window

## Common Gotchas
- **BET detection**: See `BET_FIX_EXPLANATION.md` - fixed window-overlap logic (was rejecting valid events)
- **Paths**: Code in `/burg-archive/home/$USER/phenofusion/`, data in `/burg/glab/users/.../data/`
- **Always verify paths exist** before I/O: Use `os.path.exists()` or `Path().exists()`
- **Cartopy**: Fails on headless nodes without X11 - mock in tests or use `#SBATCH -C mem768` (CPU nodes with GEOS)
- **CUDA mismatch**: Use `cuda11.8/toolkit` for new training, `cuda11.1` for legacy prediction scripts
- **Pickle compatibility**: Models trained with specific `tft-torch` version - reinstall in each job
- **Memory**: BET/BDT require 160G+ RAM; GRA/SHR can use less (80G)

## Reference Docs
- `BET_FIX_EXPLANATION.md`: BET phenology detection logic and slope verification fixes
- `docs/REFACTORING_SUMMARY.md`: Recent code improvements (modular design, error handling)
- `docs/SUBPROJECTS_GUIDE.md`: How to create analysis subprojects (gitignored experiments)
- `bashscripts/`: Canonical SLURM job templates for each PFT
- `pyproject.toml`: Dependencies and package configuration
