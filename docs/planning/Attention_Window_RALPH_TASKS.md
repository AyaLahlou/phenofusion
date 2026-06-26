# Attention Window / Legacy Effects Maps — Ralph Loop Task List

**Goal:** Produce complete, scientifically framed global maps of climate legacy effects on
phenology (SOS/EOS) for all 6 PFTs (BDT, BET, NET, NDT, GRA, SHR).

**How to use this file:** At each loop iteration, run the completion check first. If all
expected maps are present and non-trivial, stop — no wakeup. Otherwise find the first
unchecked task, execute it, update the file. Tasks that wait on SLURM jobs will tell you
to schedule a wakeup — do that and stop. Never execute more than one numbered task per
iteration.

---

## Completion check (run this first every iteration)

```bash
python3 - <<'EOF'
import os

maps_dir = "data/attention_window/maps"
expected = [
    "BDT_LegacyDepth_SOS.png",   "BDT_LegacyDepth_EOS.png",
    "BET_LegacyDepth_SOS.png",   "BET_LegacyDepth_EOS.png",
    "NET_LegacyDepth_SOS.png",   "NET_LegacyDepth_EOS.png",
    "NDT_LegacyDepth_SOS.png",   "NDT_LegacyDepth_EOS.png",
    "GRA_LegacyDepth_SOS.png",   "GRA_LegacyDepth_EOS.png",
    "SHR_LegacyDepth_SOS.png",   "SHR_LegacyDepth_EOS.png",
    "All_PFTs_LegacyDepth_SOS.png",           "All_PFTs_LegacyDepth_EOS.png",
    "All_PFTs_MultiPanel_LegacyDepth_SOS.png","All_PFTs_MultiPanel_LegacyDepth_EOS.png",
]

missing, small = [], []
for f in expected:
    path = os.path.join(maps_dir, f)
    if not os.path.exists(path):
        missing.append(f)
    elif os.path.getsize(path) < 100_000:
        small.append(f"{f} ({os.path.getsize(path)//1024} KB)")

if not missing and not small:
    print("COMPLETE — all 16 maps present and non-trivial. Stop the loop.")
else:
    if missing:
        print(f"MISSING ({len(missing)}): {', '.join(missing)}")
    if small:
        print(f"SUSPICIOUSLY SMALL: {', '.join(small)}")
    print("NOT DONE — proceed to first unchecked task.")
EOF
```

**If the script prints `COMPLETE`: stop the loop — do not schedule a wakeup.**
If it prints `NOT DONE`: continue to the task list below.

**Working directory for all shell commands:** `/burg-archive/home/al4385/phenofusion`

---

## Job ID log (fill in as jobs are submitted)

```
GRA_IDX_JOB=N/A (indices already existed, 91353 SOS / 119063 EOS)
BET_ATT_JOB=N/A (BET CSVs already existed, 18083 rows)
SHR_ATT_JOB=8431251 (COMPLETED but only 21% coverage — see SHR_SLOPE below)
GRA_ATT_JOB=8431250 (COMPLETED, 99.4% coverage)
SHR_SLOPE_IDX_JOB=8431256 (RUNNING — slope-based SHR index regeneration)
SHR_ATT_JOB_2=TBD (submit after SHR_SLOPE_IDX_JOB completes)
```

### SHR coverage diagnosis (2026-06-08)
SHR first run (job 8431251) produced only 21% valid coverage (750/3564 locations) because
strict monotonicity in save_sos_eos_indices_SHR.py rejects noisy CSIF signals.
New script save_sos_eos_indices_SHR_slope.py uses slope > 0.0001 threshold instead.
After job 8431256 completes: verify indices non-empty, resubmit get_attention_window_SHR.sh.

---

## Current state (as of 2026-06-04)

- BDT CSV: ✅ exists (`data/attention_window/BDT_SOS.csv`, 79 982 rows)
- NET CSV: ✅ exists (`data/attention_window/NET_SOS.csv`, 12 837 rows)
- NDT CSV: ✅ exists (`data/attention_window/NDT_SOS.csv`, 5 887 rows)
- BET CSV: ❌ missing — indices exist, job not yet submitted
- SHR CSV: ❌ missing — indices exist, job not yet submitted
- GRA CSV: ❌ missing — indices are empty, must regenerate first
- Maps: only BDT individual + incomplete All-PFTs (BET/GRA/SHR panels blank)

---

## Task 1 — Submit GRA SOS/EOS index generation job

GRA is the only PFT with empty index files (128 B placeholder). All input files confirmed present.

```bash
cd /burg-archive/home/al4385/phenofusion
JID=$(sbatch bashscripts/drivers/generate_sos_eos_indices_GRA.sh | awk '{print $4}')
echo "GRA_IDX_JOB=$JID"
```

After submitting, record the printed job ID in the `Job ID log` section above.

- [x] GRA index job submitted and job ID recorded in log (SKIP — indices already non-empty)

---

## Task 2 — Submit BET and SHR attention window extraction jobs

BET and SHR already have valid index files (BET: 182 293 SOS / 154 007 EOS; SHR: 11 445 / 6 137).
Submit both immediately — do not wait for Task 1.

```bash
cd /burg-archive/home/al4385/phenofusion
JID_BET=$(sbatch bashscripts/drivers/get_attention_window_BET.sh | awk '{print $4}')
JID_SHR=$(sbatch bashscripts/drivers/get_attention_window_SHR.sh | awk '{print $4}')
echo "BET_ATT_JOB=$JID_BET"
echo "SHR_ATT_JOB=$JID_SHR"
```

Record both job IDs in the log above.

- [x] BET attention window job submitted and job ID recorded (SKIP — BET CSVs already existed)
- [x] SHR attention window job submitted and job ID recorded (job 8431251)

---

## Task 3 — Wait for GRA index job, then verify indices

Look up `GRA_IDX_JOB` from the log and check its status:

```bash
GRA_IDX_JOB=<fill from log>
sacct -u al4385 --format=JobID,State,Elapsed -j $GRA_IDX_JOB --noheader
```

**If state is `RUNNING` or `PENDING`: schedule a wakeup in 1800 seconds ("waiting for GRA index job") and stop.**

If `COMPLETED`, verify the output files are non-empty:

```bash
python3 -c "
import numpy as np
sos = np.load('data/SOS_EOS_indices/GRA/SOS_pred_indices_GRA.npy')
eos = np.load('data/SOS_EOS_indices/GRA/EOS_pred_indices_GRA.npy')
print('GRA SOS:', sos.shape, '  GRA EOS:', eos.shape)
assert sos.shape[0] > 0 and eos.shape[0] > 0, 'ERROR: one or both arrays are empty'
print('OK')
"
```

If the assert fails (arrays still empty), check the error log, fix, re-submit, update the log,
then **schedule a wakeup in 1800 seconds ("re-submitted GRA index job") and stop**.

```bash
ls -t sos_eos_indices_GRA_*.err 2>/dev/null | head -1 | xargs tail -30
```

- [x] GRA index job COMPLETED and both arrays non-empty (SKIP — pre-existing: 91353 SOS, 119063 EOS)

---

## Task 4 — Submit GRA attention window extraction job

Only run after Task 3 passes (GRA indices verified non-empty).

```bash
cd /burg-archive/home/al4385/phenofusion
JID=$(sbatch bashscripts/drivers/get_attention_window_GRA.sh | awk '{print $4}')
echo "GRA_ATT_JOB=$JID"
```

Record the job ID in the log above.

- [x] GRA attention window job submitted and job ID recorded (job 8431250)

---

## Task 5 — Wait for BET, SHR, GRA attention window jobs to complete

Look up all three job IDs from the log:

```bash
BET_ATT_JOB=<fill from log>
SHR_ATT_JOB=<fill from log>
GRA_ATT_JOB=<fill from log>
sacct -u al4385 --format=JobID,JobName,State,Elapsed,MaxRSS \
  -j $BET_ATT_JOB,$SHR_ATT_JOB,$GRA_ATT_JOB --noheader
```

**If any job shows `RUNNING` or `PENDING`: schedule a wakeup in 1800 seconds ("waiting for
attention window jobs: BET/SHR/GRA") and stop.**

If any job shows `FAILED`, check its error log:
```bash
# Replace PFT and JOB_ID as needed:
ls -t attention_window_BET_*.err 2>/dev/null | head -1 | xargs tail -40
ls -t attention_window_SHR_*.err 2>/dev/null | head -1 | xargs tail -40
ls -t attention_window_GRA_*.err 2>/dev/null | head -1 | xargs tail -40
```
Common fixes:
- **OOM**: edit the SLURM script, increase `--mem-per-cpu` by 40 GB, resubmit, update log,
  then **schedule a wakeup in 1800 seconds and stop**.
- **File not found**: re-run the step-0 path checks from `Attention_Window_Maps_HPC_TASK_LIST.md`.

- [ ] All three attention window jobs show State=COMPLETED

---

## Task 6 — Verify all 6 PFT attention window CSVs

```bash
python3 - <<'EOF'
import pandas as pd, os

base = "data/attention_window"
pfts = ["BDT", "BET", "NET", "NDT", "GRA", "SHR"]
all_ok = True

for pft in pfts:
    for phase in ["SOS", "EOS"]:
        path = f"{base}/{pft}_{phase}.csv"
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            all_ok = False
            continue
        df = pd.read_csv(path)
        n = len(df)
        valid = df["window_start"].notna().sum()
        pct = 100 * valid / n if n > 0 else 0
        ws_range = f"[{df['window_start'].min():.0f}, {df['window_start'].max():.0f}]"
        status = "OK" if pct >= 90 else "WARN (low coverage)"
        print(f"{pft} {phase}: {n} rows, {valid} valid ({pct:.1f}%), ws {ws_range} — {status}")
        if pct < 90:
            all_ok = False

print()
print("ALL PASS" if all_ok else "SOME CHECKS FAILED — fix before proceeding")
EOF
```

All 12 files must be present and each must have ≥ 90% non-null `window_start` values in range [0, 335].

- [ ] All 12 CSVs verified present and ≥ 90% valid

---

## Task 7 — Edit `generate_window_maps.py` to show legacy depth

This task rewrites the script to display **legacy depth** (= 335 − window_start) instead of
raw window_start. Legacy depth answers: "how many days before the forecast horizon does the
model need to look to capture the most informative climate signal?"

Make all edits in `scripts/generate_window_maps.py`.

### Edit 7a — Add `legacy_depth` column in `load_csv_data` (after line 104)

Find this block (around line 102–106):
```python
                df = df[df["latitude"] < lat_filter]
            logger.info(f"  After lat filter: {len(df)} records")

        return df
```

Replace with:
```python
                df = df[df["latitude"] < lat_filter]
            logger.info(f"  After lat filter: {len(df)} records")

        df["legacy_depth"] = 335 - df["window_start"]
        return df
```

### Edit 7b — Switch `build_grid` to aggregate `legacy_depth` instead of `window_start`

In `build_grid` (around lines 136–142), replace all three occurrences of `"window_start"`:

```python
        df = df.dropna(subset=["window_start"])
        logger.info(f"  After dropping NaN window_start: {len(df)} rows")

        df = df.drop_duplicates(subset=["latitude", "longitude", "window_start"], keep="first")
        logger.info(f"  After dropping exact duplicates: {len(df)} rows")

        df = df.groupby(["latitude", "longitude"], as_index=False)["window_start"].median()
```

Replace with:
```python
        df = df.dropna(subset=["legacy_depth"])
        logger.info(f"  After dropping NaN legacy_depth: {len(df)} rows")

        df = df.drop_duplicates(subset=["latitude", "longitude", "legacy_depth"], keep="first")
        logger.info(f"  After dropping exact duplicates: {len(df)} rows")

        df = df.groupby(["latitude", "longitude"], as_index=False)["legacy_depth"].median()
```

Also update the grid merge log message (around line 151):
```python
        logger.info(f"  Full grid: {len(full)} cells ({full['window_start'].notna().sum()} with data)")
```
Replace with:
```python
        logger.info(f"  Full grid: {len(full)} cells ({full['legacy_depth'].notna().sum()} with data)")
```

### Edit 7c — Update `plot_map` to use `legacy_depth` and new labels

In `plot_map` (around line 185), replace:
```python
        data_2d = np.reshape(grid_df["window_start"].values, (720, 1440))
```
with:
```python
        data_2d = np.reshape(grid_df["legacy_depth"].values, (720, 1440))
```

Also change the default `colorbar_label` parameter (line 168):
```python
        colorbar_label: str = "Window start index (days)",
```
Replace with:
```python
        colorbar_label: str = "Legacy depth (days before forecast horizon)",
```

### Edit 7d — Update `generate_multi_panel_map` to use `legacy_depth`

Around line 321, replace:
```python
            all_vals = np.concatenate([
                g["window_start"].dropna().values for g in all_grids.values()
            ])
```
with:
```python
            all_vals = np.concatenate([
                g["legacy_depth"].dropna().values for g in all_grids.values()
            ])
```

Around line 336, replace:
```python
            data_2d = np.reshape(grid_df["window_start"].values, (720, 1440))
```
with:
```python
            data_2d = np.reshape(grid_df["legacy_depth"].values, (720, 1440))
```

Around line 374, replace:
```python
        cbar.set_label("Window start index (days)", fontsize=12)
```
with:
```python
        cbar.set_label("Legacy depth (days before forecast horizon)", fontsize=12)
```

Around line 376–379, replace:
```python
        fig.suptitle(
            f"Max-Attention Window Start — All PFTs ({phase})",
            fontsize=20, fontweight="bold", y=0.98,
        )
```
with:
```python
        fig.suptitle(
            f"Climate Legacy Depth — All PFTs ({phase})\n"
            f"Days before forecast horizon that capture peak attention\n"
            f"(0 = current season only;  335 = prior-year climate drives phenology)",
            fontsize=14, fontweight="bold", y=0.99,
        )
```

Around line 383, replace:
```python
            out_path = self.output_dir / f"All_PFTs_MultiPanel_WindowStart_{phase}.png"
```
with:
```python
            out_path = self.output_dir / f"All_PFTs_MultiPanel_LegacyDepth_{phase}.png"
```

### Edit 7e — Update per-PFT panel titles in `generate_multi_panel_map`

Around line 366, replace:
```python
            ax.set_title(f"{pft} — {phase}", fontsize=14, fontweight="bold")
```
with:
```python
            ax.set_title(f"{pft} — {phase}", fontsize=13, fontweight="bold")
```

### Edit 7f — Update output filenames and colorbar label in `main()`

In `main()`, around lines 477, 490, 495, 508, replace every occurrence of `"WindowStart"` in
the output name strings:

```python
                    f"{pft}_WindowStart_{phase}", **plot_kwargs,
```
→
```python
                    f"{pft}_LegacyDepth_{phase}", **plot_kwargs,
```

```python
                f"BDT_WindowStart_{phase}", **plot_kwargs,
```
→
```python
                f"BDT_LegacyDepth_{phase}", **plot_kwargs,
```

```python
                f"All_PFTs_WindowStart_{phase}", **plot_kwargs,
```
→
```python
                f"All_PFTs_LegacyDepth_{phase}", **plot_kwargs,
```

### Verify the edits with a dry-run import

```bash
cd /burg-archive/home/al4385/phenofusion
python3 -c "
import ast, sys
with open('scripts/generate_window_maps.py') as f:
    src = f.read()
ast.parse(src)
assert 'legacy_depth' in src, 'legacy_depth column not found'
assert 'WindowStart' not in src, 'old WindowStart strings still present — check edits 7d/7f'
assert 'Legacy depth' in src, 'colorbar label not updated'
print('All checks passed')
"
```

- [x] All 7 sub-edits applied and dry-run import passes

---

## Task 8 — Run `generate_window_maps.py`

```bash
cd /burg-archive/home/al4385/phenofusion
mkdir -p data/attention_window/maps

python3 scripts/generate_window_maps.py \
    --data-dir data/attention_window/ \
    --output-dir data/attention_window/maps/ \
    --projection Robinson \
    --cmap RdYlBu_r \
    --dpi 300

echo "Exit code: $?"
echo "PNG count: $(ls data/attention_window/maps/*.png 2>/dev/null | wc -l)"
ls -lh data/attention_window/maps/*.png 2>/dev/null
```

Expected output: ≥ 16 PNG files. If any PFT is still missing data, the script will log a
warning but continue — this is acceptable only for BDT band variants (no band-split CSVs exist yet).

If the script errors, check for:
- Remaining `window_start` references causing a `KeyError` (re-check Task 7 edits)
- Missing `cartopy` or other imports (run `pip install cartopy` if needed)

- [ ] Script completed without errors and ≥ 16 PNGs written

---

## Task 9 — Verify outputs and check scientific plausibility

```bash
python3 - <<'EOF'
import pandas as pd, numpy as np, os

base = "data/attention_window"
pfts = ["BDT", "BET", "NET", "NDT", "GRA", "SHR"]

print("=== Legacy depth summary (should vary meaningfully across PFTs) ===")
for pft in pfts:
    for phase in ["SOS", "EOS"]:
        path = f"{base}/{pft}_{phase}.csv"
        if not os.path.exists(path):
            print(f"  {pft} {phase}: MISSING CSV")
            continue
        df = pd.read_csv(path)
        ld = 335 - df["window_start"].dropna()
        print(f"  {pft} {phase}: median={ld.median():.0f} d  IQR=[{ld.quantile(0.25):.0f}, {ld.quantile(0.75):.0f}]  N={len(ld)}")

print()
print("=== Map file sizes (< 100 KB likely blank) ===")
maps_dir = f"{base}/maps"
for f in sorted(os.listdir(maps_dir)):
    if f.endswith(".png"):
        size_kb = os.path.getsize(f"{maps_dir}/{f}") / 1024
        flag = " ← SUSPICIOUSLY SMALL" if size_kb < 100 else ""
        print(f"  {f}: {size_kb:.0f} KB{flag}")
EOF
```

**Scientific plausibility checks:**
- NET and NDT (boreal conifers/larch) should show **higher** median legacy depth for SOS
  than BET (tropical) — snowpack and prior-winter temperature drive boreal budburst.
- BET EOS legacy depth expected to be low (tropical systems respond to recent conditions).
- GRA and SHR should show intermediate legacy depth, with latitudinal gradients visible.
- No map file should be < 100 KB (would indicate blank/empty grid).

- [ ] All map files ≥ 100 KB and legacy depth medians differ meaningfully across PFTs

---

## Task 10 — Done

```bash
echo "All tasks complete."
ls -lh data/attention_window/maps/All_PFTs_MultiPanel_LegacyDepth_SOS.png
ls -lh data/attention_window/maps/All_PFTs_MultiPanel_LegacyDepth_EOS.png
```

- [ ] Final multi-panel figures confirmed present

---

## Appendix — if a SLURM job fails

**OOM (exit 1, `oom-kill` in .err):**
```bash
# Example for BET — increase mem and resubmit
sed -i 's/--mem-per-cpu=160G/--mem-per-cpu=200G/' bashscripts/drivers/get_attention_window_BET.sh
JID=$(sbatch bashscripts/drivers/get_attention_window_BET.sh | awk '{print $4}')
echo "new BET_ATT_JOB=$JID"   # update the log section above
```

**Empty index file (for GRA regeneration failure):**
```bash
# Check the GRA analysisdf exists and is non-empty
ls -lh /burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/GRA_analysisdf_20251106
python3 -c "import pandas as pd; df=pd.read_csv('/burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/GRA_analysisdf_20251106'); print(df.shape, df.columns.tolist())"
```
