# Paper Changelog — Phenofusion → Nature Climate Change

Tracks substantive changes to `nature_draft.tex`, decisions, and progress milestones.
Format: newest entries at top.

---

## 2026-06-26

### Results: Forecasting Performance
- Rewrote §Forecasting Performance to include all 6 PFTs (was missing GRA and SHR)
- GRA: R²_overall=0.99, R²_anomaly=0.24; SHR: R²_overall=0.98, R²_anomaly=0.67
- Added interpretation: SHR anomaly skill (0.67) rivals BDT (0.69); GRA low due to pulse-precipitation stochasticity
- Added explicit citation of process-based model failure at interannual anomalies (Richardson et al. 2012, Peano et al. 2021)

### Results: Senescence / EOS drivers (new section)
- Filled in the previously empty §Senescence subsubsection
- Key patterns: photoperiod more prominent at EOS than SOS (50–70% in boreal); temperature still dominant in temperate; water less prominent at EOS than SOS
- PFT-specific: NDT strongly photoperiod-controlled at EOS; BET balanced temperature–radiation; GRA/SHR spatially heterogeneous

### Results: Climate Legacy Effects (rewrite)
- Replaced 2-paragraph vague stub with 4 quantitative paragraphs
- BET and NDT: median attention window_start ≈ 24–27 → model attends to conditions ~12 months prior (strong annual memory)
- NET SOS: median window_start ≈ 81 → ~9 months prior (previous summer conditions)
- BDT, GRA, SHR: median window_start = 335 → current-season conditions only, no legacy
- Added `fig:legacy` figure: `data/attention_window/maps/All_PFTs_LegacyDepth_SOS.png`

### Results: Long-term Shifts in Climate Sensitivity (rewrite)
- Replaced qualitative paragraph with PFT-by-PFT quantified Δ values from `data/sensitivity_figures/AllPFT_summary_heatmap.png`
- BDT SOS: Δwater=+0.014 (54% pixels sig pos), Δsolar=−0.013 (54% sig neg)
- SHR SOS: Δwater=+0.012, Δsolar=−0.011
- BET: 50% pixels show significant water increase at SOS, 40% at EOS
- GRA SOS anomalous: Δwater=−0.005 (not sig); possible redistribution of moisture stress seasonality
- Temperature: small, spatially heterogeneous changes across all PFTs
- Added `fig:sensitivity` figure: `data/sensitivity_figures/AllPFT_combined_delta_panel.png`

### Discussion: PFT abbreviation fix
- Replaced DBT/EBT/DNT/ENT → BDT/BET/NDT/NET throughout Discussion section

### Minor fixes
- Removed stray `\` command before `\bibliography`

---

## 2026-06-25 (previous session)

### Repository cleanup (branch: paper/cleanup)
- Moved SLURM logs to `logs/`; removed stray `.err` files from git tracking
- Updated `.gitignore`: added logs/, debugging/, scratch notebooks, sensitivity data dirs
- Removed duplicate `scripts/sensitivity_shifts.py` (canonical is `src/phenofusion/sensitivity_shifts.py`)
- Fixed `CLAUDE.md` stale note about sensitivity_shifts.py location

### Pre-commit hook fix
- Removed `black` from `.pre-commit-config.yaml` (was conflicting with `ruff-format`, causing infinite reformat loop)

### Ruff lint fixes
- `sensitivity_shifts.py`: renamed ambiguous variable `l` → `late` (E741) in three places
- `notebooks/clean/`: fixed F811 duplicate method, F841 unused variables

### Journal decision
- **Target: Nature Climate Change** (not Nature flagship)
- Reasoning: Li et al. 2022 Nat Comms already showed water↑ for GPP/NDVI; driver shift not wholly novel; anomaly R²=0.69 is the stronger novelty claim; NCC publishes global ML×vegetation attribution

### Literature review (13 papers in `paper/Litterature/`)
- PhenoFormer (Garnot 2025): country-scale Switzerland, nRMSE on DOY — different scale/metric, not direct competitor
- MIRANDA (Jiang 2026): R²≈0.49–0.50 species-level DOY in Switzerland — same limitation
- Li et al. 2022 Nat Comms: water sensitivity increase already shown for NDVI/GPP — our novelty: phenological timing specifically
- Identified `tasks.md` open questions 1–5 (see `paper/tasks.md`)

### `paper/tasks.md` created
- Phase 0–4 task breakdown with status tracking
- Blocking items (1A metrics, 1B driver maps, 1C shift quantification) identified

### Data exploration
- Computed attention window statistics from `data/attention_window/{PFT}_{SOS|EOS}.csv`
- Read `outputs/metrics/metrics_GRA.png` and `metrics_SHR.png` as images for R² values
- Read `data/sensitivity_figures/summary_table.csv` for pixel-weighted Δ driver statistics
