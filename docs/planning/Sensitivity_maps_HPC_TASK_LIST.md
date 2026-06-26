# HPC Task List: Sensitivity-Shift Figure Generation

Instructions for Claude Code running on the Columbia HPC (Burg) cluster.
Execute each numbered task in order. Check the PASS criterion before proceeding.
Record the completion of each task in the log.

**Working directory for all commands:** `/burg-archive/home/al4385/phenofusion`

---

## 0. Setup — verify paths and create output directories

```bash
cd /burg-archive/home/al4385/phenofusion

SENSITIVITY_DATA="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
SENSITIVITY_FIGS="/burg-archive/home/al4385/phenofusion/data/sensitivity_figures"
mkdir -p "$SENSITIVITY_DATA" "$SENSITIVITY_FIGS"

# Confirm all input data files exist before submitting any jobs
echo "=== Checking input files ==="
ls -lh /burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/BET_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/BET_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/NET_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/NET_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/NDT_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/NDT_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/GRA_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/GRA_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/SHR_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/SHR_1982_2021.pkl
```

**PASS:** All 12 files listed without errors.

---

## 1. Check SOS/EOS index files

```bash
echo "=== SOS/EOS index inventory ==="
for f in \
  /burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/SOS_pred_indices_BDT_1.npy \
  /burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/EOS_pred_indices_BDT_1.npy \
  /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NET/SOS_pred_indices_NET.npy \
  /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NET/EOS_pred_indices_NET.npy \
  /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/SOS_pred_indices_NDT.npy \
  /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/EOS_pred_indices_NDT.npy \
  /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/SHR/SOS_pred_indices_SHR.npy \
  /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/SHR/EOS_pred_indices_SHR.npy; do
  python3 -c "import numpy as np; a=np.load('$f'); print('$(basename $f):', a.shape)"
done
```

**Expected results:**
- `SOS_pred_indices_BDT_1.npy`: shape `(163317,)` — full merged BDT dataset
- NET SOS/EOS: shapes ~`(140651,)` / `(...)` — large arrays
- NDT SOS/EOS: shapes ~`(44180,)` / `(...)` — mid-size arrays
- SHR SOS/EOS: shapes ~`(11445,)` / `(...)` — smaller arrays

**Note:** BET and GRA index files are empty (shape `(0,)`) — the SLURM scripts for BET and GRA use `--detect-csif` automatically. Do NOT try to pass their `.npy` files.

**PASS:** BDT, NET, NDT, SHR arrays are non-empty.

---

## 2–7. Per-PFT epoch driver extraction (SLURM jobs)

SLURM scripts are at `bashscripts/analysis/sensitivity_*.sh`. Submit all six in
one go, then monitor until all complete. Output CSVs land in
`data/sensitivity_data/`.

```bash
cd /burg-archive/home/al4385/phenofusion

JID_BDT=$(sbatch bashscripts/analysis/sensitivity_BDT.sh | awk '{print $4}')
JID_BET=$(sbatch bashscripts/analysis/sensitivity_BET.sh | awk '{print $4}')
JID_NET=$(sbatch bashscripts/analysis/sensitivity_NET.sh | awk '{print $4}')
JID_NDT=$(sbatch bashscripts/analysis/sensitivity_NDT.sh | awk '{print $4}')
JID_GRA=$(sbatch bashscripts/analysis/sensitivity_GRA.sh | awk '{print $4}')
JID_SHR=$(sbatch bashscripts/analysis/sensitivity_SHR.sh | awk '{print $4}')

echo "Submitted: BDT=$JID_BDT BET=$JID_BET NET=$JID_NET NDT=$JID_NDT GRA=$JID_GRA SHR=$JID_SHR"
```

### 2. BDT — Broadleaf Deciduous Trees (large, ~170 GB pred file)
Script: `bashscripts/analysis/sensitivity_BDT.sh` — 2 CPUs × 200 GB, mem768 node, 12 h
Uses `--sos-indices` / `--eos-indices` (163317 events, matched to merged BDT pkl).

### 3. BET — Broadleaf Evergreen Trees (large, ~201 GB pred file)
Script: `bashscripts/analysis/sensitivity_BET.sh` — 2 CPUs × 200 GB, mem768 node, 12 h
Uses `--detect-csif` (BET index files are empty).

### 4. NET — Needleleaf Evergreen Trees (~143 GB pred file)
Script: `bashscripts/analysis/sensitivity_NET.sh` — 2 CPUs × 160 GB, mem768 node, 12 h

### 5. NDT — Needleleaf Deciduous Trees (~66 GB pred file)
Script: `bashscripts/analysis/sensitivity_NDT.sh` — 2 CPUs × 80 GB, 8 h

### 6. GRA — Grasses (~133 GB pred file)
Script: `bashscripts/analysis/sensitivity_GRA.sh` — 2 CPUs × 160 GB, mem768 node, 12 h
Uses `--detect-csif` (GRA index files are empty).

### 7. SHR — Shrubs (~22 GB pred file)
Script: `bashscripts/analysis/sensitivity_SHR.sh` — 2 CPUs × 60 GB, 6 h

---

## 8. Monitor jobs until all complete

```bash
# Poll until all submitted jobs finish
watch -n 60 "squeue -u al4385 --format='%.10i %.15j %.8T %.10M %.6D' | grep -E 'sens|JOBID'"

# Or check once:
squeue -u al4385 | grep sens

# When done, check exit codes:
sacct -u al4385 --format=JobID,JobName,State,Elapsed,MaxRSS -j ${JID_BDT},${JID_BET},${JID_NET},${JID_NDT},${JID_GRA},${JID_SHR}
```

If any job FAILED, check its `.err` file (written to the phenofusion repo root):
```bash
tail -50 sensitivity_BDT_${JID_BDT}.err   # replace PFT/JobID as needed
```

Common failure causes:
- **OOM (out of memory)**: increase `--mem-per-cpu` in the SLURM script by 50 GB and resubmit.
- **No SOS/EOS events detected**: CSIF values may be near-constant. Check with:
  ```bash
  python3 -c "import pickle; d=pickle.load(open('PATH.pkl','rb')); t=d['data_sets']['test']['target']; print(t.min(), t.max(), t.std())"
  ```
  If `std ≈ 0`, the wrong data file was loaded.
- **File not found**: re-run the step-0 path checks; a file may have been moved.

**PASS:** All 6 jobs show State=COMPLETED in sacct.

---

## 9. Verify CSV outputs

```bash
SENSITIVITY_DATA="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"

echo "=== File inventory ==="
ls -lh "$SENSITIVITY_DATA"/*.csv 2>/dev/null | awk '{print $5, $9}'

echo ""
echo "=== Year coverage and row counts per PFT ==="
for f in "$SENSITIVITY_DATA"/*_yearly_drivers.csv; do
  pft=$(basename "$f" _yearly_drivers.csv)
  n=$(wc -l < "$f")
  yr_min=$(awk -F, 'NR>1{print $5}' "$f" | sort -n | head -1)
  yr_max=$(awk -F, 'NR>1{print $5}' "$f" | sort -n | tail -1)
  phases=$(awk -F, 'NR>1{print $4}' "$f" | sort -u | tr '\n' '/')
  echo "$pft: $((n-1)) rows, years $yr_min–$yr_max, phases $phases"
done

echo ""
echo "=== Driver value ranges (first CSV) ==="
python3 -c "
import pandas as pd, glob, sys
files = sorted(glob.glob('$SENSITIVITY_DATA/*_yearly_drivers.csv'))
if not files: sys.exit('no files found')
df = pd.read_csv(files[0])
print(df[['hist_tmin','hist_tmax','hist_rad','hist_precip','hist_photo','hist_sm']].describe().round(4))
"
```

**PASS criteria:**
- All 6 PFT CSVs present: `BDT_yearly_drivers.csv`, `BET_yearly_drivers.csv`, `NET_yearly_drivers.csv`, `NDT_yearly_drivers.csv`, `GRA_yearly_drivers.csv`, `SHR_yearly_drivers.csv`
- Year range is 1982–2021 (sparse edges are OK).
- Both `SOS` and `EOS` phases appear for each PFT (BET may only show EOS — acceptable).
- Driver values all in [0, 1] — no negatives, no values > 1.

---

## 10. Generate all figures

Run locally (no SLURM needed — reads small CSVs, runs in memory):

```bash
cd /burg-archive/home/al4385/phenofusion

SENSITIVITY_DATA="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
SENSITIVITY_FIGS="/burg-archive/home/al4385/phenofusion/data/sensitivity_figures"
mkdir -p "$SENSITIVITY_FIGS"

python scripts/run_sensitivity_shifts.py \
  --data-dir "$SENSITIVITY_DATA" \
  --output-dir "$SENSITIVITY_FIGS"
```

Monitor progress (in another terminal or after the run):
```bash
ls "$SENSITIVITY_FIGS"/*.png | wc -l
```

For a quick preview first (fewer bootstrap samples, skip tier 3):
```bash
python scripts/run_sensitivity_shifts.py \
  --data-dir "$SENSITIVITY_DATA" \
  --output-dir "$SENSITIVITY_FIGS" \
  --bootstrap-B 50 --skip-tier 3
```

**Expected output count:**
- 6 PFTs × 2 phases × 6 per-PFT figures = 72
- 2 phases × 4 mosaic figures = 8
- 1 summary heatmap = 1
- **≈ 81 PNGs total** + `summary_table.csv`

**PASS:** At least 70 PNG files in `data/sensitivity_figures/` and no Python tracebacks in the run output.

---

## File naming quick-reference

| Output file | Tier | What to look for |
|---|---|---|
| `{PFT}_{phase}_delta_per_driver.png` | 1 | Signed Δ, stipple = significant pixels |
| `{PFT}_{phase}_dominance_flip.png` | 1 | Vivid off-diagonal = biome switched dominant driver |
| `AllPFT_{phase}_delta_{g}.png` | 1 | Global Δ hotspots across all PFTs |
| `AllPFT_{phase}_dominance_flip.png` | 1 | Global driver-flip map |
| `{PFT}_{phase}_zonal_profile.png` | 2 | Latitude gradient of Δ sensitivity |
| `{PFT}_{phase}_magnitude.png` | 2 | \|Δ\| hotspot map |
| `AllPFT_summary_heatmap.png` | 2 | One-figure paper summary |
| `{PFT}_{phase}_dominance_margin.png` | 3 | Is dominant driver becoming more decisive? |
| `{PFT}_{phase}_decadal_trajectory.png` | 3 | Rolling-decade evolution heatmap |
| `summary_table.csv` | — | Data for paper tables |
