# HPC Task List: Attention Window ("Legacy Effects") Map Generation

Instructions for Claude Code running on the Columbia HPC (Burg) cluster.
Execute each numbered task in order. Check the PASS criterion before proceeding.
Record the completion of each task in the log.

**Working directory for all commands:** `/burg-archive/home/al4385/phenofusion`

**Scientific framing:** The attention window captures *when* the TFT model looks back in the
historical record to make its phenology forecast. A window starting near index 0 means the
model draws on conditions ~1 year before the forecast — a strong **legacy effect** of prior-season
climate. A window near index 335 means only the most recent ~30 days matter — no legacy.
The maps answer: *"Where on Earth does prior-year climate drive phenology onset/end, and where
do only current-season conditions matter?"*

---

## State at task-list creation (2026-06-04)

| PFT | SOS/EOS indices | Attention window CSV | Individual map | In All-PFTs map |
|-----|----------------|---------------------|----------------|-----------------|
| BDT | ✅ `debugging/debugging_subset_data/SOS_pred_indices_BDT_1.npy` (163317) | ✅ `BDT_SOS.csv` / `BDT_EOS.csv` (79 982 rows, 9.5% missing) | ✅ `maps/BDT_WindowStart_*.png` | ✅ (partial) |
| NET | ✅ `data/SOS_EOS_indices/NET/` (SOS: 140651, EOS: 154781) | ✅ `NET_SOS.csv` / `NET_EOS.csv` (12 837 rows each) | ❌ | partial |
| NDT | ✅ `data/SOS_EOS_indices/NDT/` (SOS: 44180, EOS: 63347) | ✅ `NDT_SOS.csv` / `NDT_EOS.csv` (5 887 rows each) | ❌ | partial |
| BET | ✅ `data/SOS_EOS_indices/BET/` (SOS: 182293, EOS: 154007) | ❌ missing | ❌ | ❌ |
| SHR | ✅ `data/SOS_EOS_indices/SHR/` (SOS: 11445, EOS: 6137) | ❌ missing | ❌ | ❌ |
| GRA | ❌ empty (128 B placeholder) | ❌ missing | ❌ | ❌ |

**All-PFTs maps** (`maps/All_PFTs_*.png`) exist but contain only BDT + NET + NDT.
The 2×3 multi-panel figures also exist but have blank panels for BET, SHR, GRA.

---

## 0. Setup — verify paths

```bash
cd /burg-archive/home/al4385/phenofusion

echo "=== Prediction & data files ==="
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/BET_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/BET_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/GRA_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/GRA_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/SHR_1982_2021.pkl
ls -lh /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/SHR_1982_2021.pkl

echo "=== Coordinate files ==="
ls -lh /burg-archive/glab/users/al4385/data/coordinates/BET.parquet
ls -lh /burg-archive/glab/users/al4385/data/coordinates/GRA.parquet
ls -lh /burg-archive/glab/users/al4385/data/coordinates/SHR.parquet

echo "=== BET index files (must be non-empty) ==="
python3 -c "import numpy as np; a=np.load('data/SOS_EOS_indices/BET/SOS_pred_indices_BET.npy'); print('BET SOS:', a.shape)"
python3 -c "import numpy as np; a=np.load('data/SOS_EOS_indices/BET/EOS_pred_indices_BET.npy'); print('BET EOS:', a.shape)"

echo "=== SHR index files (must be non-empty) ==="
python3 -c "import numpy as np; a=np.load('data/SOS_EOS_indices/SHR/SOS_pred_indices_SHR.npy'); print('SHR SOS:', a.shape)"
python3 -c "import numpy as np; a=np.load('data/SOS_EOS_indices/SHR/EOS_pred_indices_SHR.npy'); print('SHR EOS:', a.shape)"

echo "=== GRA analysisdf (needed for index generation) ==="
ls -lh /burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/GRA_analysisdf_20251106
ls -lh debugging/debugging_subset_data/sos_eos_GRA_merged_smoothed.json
```

**PASS:** All 6 pkl files listed without errors; BET and SHR index shapes are non-zero.

---

## 1. Generate GRA SOS/EOS indices (SLURM)

GRA is the only PFT whose index files are empty. Submit the generation job:

```bash
JID_GRA_IDX=$(sbatch bashscripts/drivers/generate_sos_eos_indices_GRA.sh | awk '{print $4}')
echo "GRA index job: $JID_GRA_IDX"
```

Script: `bashscripts/drivers/generate_sos_eos_indices_GRA.sh`
- Calls `debugging/save_sos_eos_indices_GRA.py`
- Reads: `GRA_analysisdf_20251106` + `sos_eos_GRA_merged_smoothed.json` + `GRA_1982_2021.pkl`
- Writes: `data/SOS_EOS_indices/GRA/SOS_pred_indices_GRA.npy` and `EOS_pred_indices_GRA.npy`
- Memory: 64 GB, 1 h, `short` partition

Monitor:
```bash
squeue -u al4385 | grep sos_eos_GRA
```

Verify after completion:
```bash
python3 -c "import numpy as np; a=np.load('data/SOS_EOS_indices/GRA/SOS_pred_indices_GRA.npy'); print('GRA SOS:', a.shape)"
python3 -c "import numpy as np; a=np.load('data/SOS_EOS_indices/GRA/EOS_pred_indices_GRA.npy'); print('GRA EOS:', a.shape)"
```

**PASS:** Both GRA index arrays have shape `(N,)` with N > 0.

---

## 2. Submit BET and SHR attention window extraction (SLURM)

BET and SHR already have valid index files — submit immediately (do not wait for task 1):

```bash
JID_BET=$(sbatch bashscripts/drivers/get_attention_window_BET.sh | awk '{print $4}')
JID_SHR=$(sbatch bashscripts/drivers/get_attention_window_SHR.sh | awk '{print $4}')
echo "BET attention job: $JID_BET   SHR attention job: $JID_SHR"
```

Script details:
- **BET**: 160 GB per CPU, 3 h, `mem768` partition. Reads ~216 GB pred file.
- **SHR**: 160 GB per CPU, 3 h, `mem768` partition. Reads ~22 GB pred file.

Both scripts write `{PFT}_SOS.csv` and `{PFT}_EOS.csv` to `data/attention_window/`.

---

## 3. Submit GRA attention window extraction (after task 1 PASS)

Once GRA index files are verified non-empty:

```bash
JID_GRA=$(sbatch bashscripts/drivers/get_attention_window_GRA.sh | awk '{print $4}')
echo "GRA attention job: $JID_GRA"
```

Script: 160 GB per CPU, 3 h, `mem768` partition. Reads ~133 GB pred file.

---

## 4. Monitor all jobs until complete

```bash
watch -n 60 "squeue -u al4385 --format='%.10i %.20j %.8T %.10M' | grep -E 'attention|JOBID'"

# When done, check for errors:
sacct -u al4385 --format=JobID,JobName,State,Elapsed,MaxRSS \
  -j ${JID_BET},${JID_SHR},${JID_GRA}
```

If a job **FAILED**:
- **OOM**: Increase `--mem-per-cpu` by 40 GB in the SLURM script and resubmit.
- **File not found**: Re-run step 0 path checks.
- **Empty indices**: Load the `.npy` in Python and check shape; re-run task 1 if GRA is still empty.

**PASS:** All three jobs show `State=COMPLETED`.

---

## 5. Verify CSV outputs

```bash
echo "=== File inventory ==="
ls -lh data/attention_window/*.csv

echo ""
echo "=== Row counts and window_start completeness ==="
for pft in BDT BET NET NDT GRA SHR; do
  for phase in SOS EOS; do
    f="data/attention_window/${pft}_${phase}.csv"
    if [ -f "$f" ]; then
      python3 -c "
import pandas as pd
df = pd.read_csv('$f')
n_total = len(df)
n_valid = df['window_start'].notna().sum()
ws_min = df['window_start'].min()
ws_max = df['window_start'].max()
print(f'${pft} ${phase}: {n_total} rows, {n_valid} valid window_start ({100*n_valid/n_total:.1f}%), range [{ws_min:.0f}, {ws_max:.0f}]')
"
    else
      echo "${pft} ${phase}: MISSING"
    fi
  done
done
```

**PASS criteria:**
- All 12 CSVs present (6 PFTs × 2 phases).
- Each has ≥ 90% non-null `window_start` values.
- `window_start` values in range [0, 335].

---

## 6. Improve `generate_window_maps.py` for scientific output

The current script plots raw 0–335 index values. For a scientifically meaningful "legacy effects"
figure, convert to **legacy depth in days** before regenerating maps.

**Definition:**
```
legacy_depth = 365 - window_start - 30
```
- `legacy_depth = 335` → model looks back at conditions from ~335 days before the forecast
  window = **strong legacy effect** (prior growing season or prior year)
- `legacy_depth = 0` → model only uses the final 30 days before the forecast = **no legacy**

Edit `scripts/generate_window_maps.py` to:

1. **After loading each CSV**, add a derived column:
   ```python
   df["legacy_depth"] = 335 - df["window_start"]
   ```
   Use `legacy_depth` instead of `window_start` for all grid mapping and plotting.

2. **Change the colormap** from `plasma` to `RdYlBu_r` (blue = strong legacy, red = no legacy)
   so that the scientific interpretation is visually intuitive.

3. **Update colorbar labels** to read:
   - Tick at 0: "0 d (current season only)"
   - Tick at 167: "~6 months"
   - Tick at 335: "335 d (prior year)"
   - Colorbar title: "Legacy depth (days before forecast)"

4. **Update the figure titles** to read e.g. `"BDT — SOS Legacy Depth"` instead of
   `"BDT — Window Start SOS"`.

5. **Update output filenames** from `BDT_WindowStart_SOS.png` to `BDT_LegacyDepth_SOS.png`
   (change the `f"{pft}_WindowStart_{phase}"` format strings).

6. **Keep the scientific story in the multi-panel title**: add a subtitle such as
   `"How far back in time does the TFT model look to forecast phenology?\n(Legacy depth = 0: only recent weeks matter; 335 days: prior-year climate matters)"`

After editing, run:
```bash
python scripts/generate_window_maps.py \
  --data-dir data/attention_window/ \
  --output-dir data/attention_window/maps/ \
  --projection Robinson \
  --cmap RdYlBu_r \
  --dpi 300
```
and visually inspect one output PNG before proceeding.

**PASS:** `data/attention_window/maps/BDT_LegacyDepth_SOS.png` exists, colorbar reads
"Legacy depth (days before forecast)", and the map is not blank/uniform.

---

## 7. Generate all maps

Run the full map generation (reads all 12 CSVs, writes all output PNGs):

```bash
mkdir -p data/attention_window/maps/

python scripts/generate_window_maps.py \
    --data-dir data/attention_window/ \
    --output-dir data/attention_window/maps/ \
    --projection Robinson \
    --cmap RdYlBu_r \
    --dpi 300

echo "=== Output count ==="
ls data/attention_window/maps/*.png | wc -l
ls data/attention_window/maps/*.png
```

**Expected outputs:**

| File | Description |
|------|-------------|
| `BDT_LegacyDepth_SOS.png` | BDT SOS legacy depth map |
| `BDT_LegacyDepth_EOS.png` | BDT EOS legacy depth map |
| `BET_LegacyDepth_SOS.png` | BET — tropical forests |
| `BET_LegacyDepth_EOS.png` | |
| `NET_LegacyDepth_SOS.png` | NET — boreal evergreens |
| `NET_LegacyDepth_EOS.png` | |
| `NDT_LegacyDepth_SOS.png` | NDT — larch belt |
| `NDT_LegacyDepth_EOS.png` | |
| `GRA_LegacyDepth_SOS.png` | GRA — grasslands |
| `GRA_LegacyDepth_EOS.png` | |
| `SHR_LegacyDepth_SOS.png` | SHR — shrublands |
| `SHR_LegacyDepth_EOS.png` | |
| `All_PFTs_LegacyDepth_SOS.png` | All-PFTs single combined map |
| `All_PFTs_LegacyDepth_EOS.png` | |
| `All_PFTs_MultiPanel_LegacyDepth_SOS.png` | 2×3 panel figure (paper-ready) |
| `All_PFTs_MultiPanel_LegacyDepth_EOS.png` | |

**PASS:** At least 16 PNGs present. Multi-panel figures show 6 non-blank panels, each with
visible spatial gradients and a shared colorbar.

---

## 8. Visual quality check

Open the 2×3 multi-panel figures and verify each panel:

```bash
# Check file sizes — a mostly blank map will be < 100 KB
ls -lh data/attention_window/maps/All_PFTs_MultiPanel_LegacyDepth_SOS.png
ls -lh data/attention_window/maps/All_PFTs_MultiPanel_LegacyDepth_EOS.png

# Spot-check per-PFT range
for pft in BDT BET NET NDT GRA SHR; do
  python3 -c "
import pandas as pd, numpy as np
sos = pd.read_csv('data/attention_window/${pft}_SOS.csv')
eos = pd.read_csv('data/attention_window/${pft}_EOS.csv')
for phase, df in [('SOS', sos), ('EOS', eos)]:
    ld = 335 - df['window_start'].dropna()
    print(f'${pft} {phase}: median legacy={ld.median():.0f} d, IQR=[{ld.quantile(0.25):.0f}, {ld.quantile(0.75):.0f}]')
"
done
```

**What to look for scientifically:**
- **BET** (tropical forests): expect low legacy depth for EOS (no winter dormancy); SOS may
  show moderate legacy tied to dry-season length.
- **NDT/NET** (boreal): expect high legacy depth for SOS — snowpack and prior-winter temperature
  are known to control budburst up to 6–12 months in advance.
- **GRA** (grasslands): expect moderate legacy; SOS driven by winter precipitation legacy
  in semi-arid systems, but short-memory in temperate grasslands.
- **SHR**: similar to GRA but with greater arctic/subarctic representation → expect higher
  legacy depth near 60–70°N.
- **Spatial contrast within PFTs**: if legacy depth varies strongly by latitude or continent,
  that is a scientifically interesting result worth highlighting.

**PASS:** No panel is all one color. Boreal panels (NET, NDT) show higher median legacy
depth than tropical panels (BET). Maps are free of obvious artefacts (stripes, random noise).

---

## 9. Optional — upgrade BDT to current prediction files

The existing BDT CSV was generated from the *older* `predictions_40years/` pkl (TFT_30 format),
not the current `pred_40year_moresamples/` pkl. This means BDT is not fully consistent with
the other PFTs.

To regenerate BDT using the four-band current predictions, four per-band SLURM jobs would be
needed (one per latitude band: 50_90, 50_20, -20_20, -20_-60). This requires:
1. New per-band SOS/EOS index `.npy` files (currently only `BDT_1.npy` for the merged old pred).
2. New per-band bash scripts (`get_attention_window_BDT_50_90.sh`, etc.), analogous to the
   existing `drivers_BDT_5090.sh` pattern.
3. `generate_window_maps.py` already supports `BDT_5020`, `BDT_5090`, etc. — it merges them
   automatically when per-band CSVs are present.

**This step is optional for a complete 6-PFT figure.** The existing BDT CSV (79 982 rows,
merged) is sufficient for a first complete version. Upgrade if consistency across PFTs becomes
a priority.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `window_start` all NaN | Index file loaded is empty (`shape=(0,)`) | Re-check index path; regenerate if needed |
| Job OOM | Pred pkl too large for allocated memory | Increase `--mem-per-cpu` by 40 GB and resubmit |
| Blank panel in multi-panel figure | CSV missing or all-NaN for that PFT/phase | Re-check step 5 verification |
| Colorbar spans 0–335 instead of 0–335 legacy | `legacy_depth` column not created | Verify step 6 edit was saved before running step 7 |
| BET SOS looks identical to EOS | BET phenology detection not distinguishing phases | Inspect `BET_SOS.csv` vs `BET_EOS.csv` window_start distributions |
