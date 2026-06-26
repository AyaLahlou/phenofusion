# Paper Tasks — Phenofusion → Nature submission

**Target:** Nature / Nature Climate Change / Nature Geoscience (TBD after lit review)
**Deadline:** ~1 week
**Core claim:** TFT trained on LCSIF + ERA5 predicts global vegetation phenology (R² ≈ 0.99 overall, R² = 0.69 interannual anomalies for BDT); attention weights reveal a 1982–2022 shift — water availability increasingly dominant, temperature declining.

---

## Status legend
- [ ] TODO
- [~] In progress
- [x] Done

---

## PHASE 0 — Unblocks everything else

- [ ] **Share literature folder path** so key prior work can be reviewed before finalizing framing
- [ ] **Literature review** (read top ~10 papers in folder): identify what anomaly R² values existing models achieve, whether water↑/temperature↓ shift has been shown at global scale, how other ML phenology papers are framed
- [ ] **Decide journal target**: Nature flagship (needs driver shift to be paradigm-shifting relative to prior attribution literature) vs. Nature Climate Change (safer fit — climate×vegetation attribution at global scale) vs. Nature Geoscience

---

## PHASE 1 — Blocking: results must be complete before writing

### 1A. Metrics for all PFTs

- [x] GRA overall/seasonal/anomaly metrics (figure exists: `outputs/metrics/metrics_GRA.png`)
- [x] SHR overall/seasonal/anomaly metrics (figure exists: `outputs/metrics/metrics_SHR.png`)
- [ ] **BDT anomaly R² = 0.69 — reproduce from `metrics_tft.py`** (confirm number, get RMSE + BIAS too)
- [ ] **BET anomaly R²** — run `metrics_tft.py` via SLURM (pred: `BET_1982_2021.pkl`, data: `BET_1982_2021.pkl`)
- [ ] **NET anomaly R²** — run `metrics_tft.py` via SLURM
- [ ] **NDT anomaly R²** — run `metrics_tft.py` via SLURM
- [ ] **Write a metrics aggregation script** (`scripts/collect_metrics.py` or similar) that runs all PFTs and outputs a single CSV table with columns: `PFT, R2_overall, R2_across_site, R2_seasonal, R2_anomaly, RMSE_anomaly, BIAS_anomaly` — one row per PFT. Currently `metrics_tft.py` only prints to stdout; need to capture and tabulate.
- [ ] **Benchmark comparison table**: retrieve anomaly R² from 4GST, PhenoFormer, ThermalTime predictions (data in `/burg/glab/users/al4385/data/4GST/`, `phenoformer/`, `pyphenology_benchmark/`) for BDT at minimum — this is the number that makes the anomaly result meaningful

### 1B. Driver attribution completeness

- [x] Driver attribution maps: BDT (combined 4-band), BET, NDT, NET (`outputs/driver_maps/`)
- [ ] **GRA and SHR driver maps** — check whether `All_PFTs_SOS.png` includes them; if not, re-run `run_driver_maps.py` with GRA/SHR data from `ms7073/analysis/driversdata/oversampling/driver_maps_oversampling_20251204/`
- [ ] **Verify GRA/SHR are in All_PFTs_* combined maps** — open those PNGs and confirm coverage

### 1C. Driver shift quantification

- [x] Bootstrap Δ maps with 95% CI computed for all PFTs (`data/sensitivity_figures/`, 83 files)
- [x] `summary_table.csv` exists with per-PFT mean_dtemp, mean_dsol, mean_dp + fraction significant
- [ ] **Extract and report headline numbers cleanly**: from `summary_table.csv`, compute weighted global-mean Δ per driver (weighted by n_pixels); report as "globally, water attribution weight increased by X±Y percentage points while temperature declined by Z±W points (fraction of pixels significant: F%)"
- [ ] **Assess significance of driver shift**: many `frac_sig_neg_dtemp` values in summary table are ~0.4–0.5, meaning ~half of pixels show significant temp decline. Flag this honestly — is the shift robust or is there large spatial heterogeneity that needs to be foregrounded?
- [ ] **Statistical significance for driver attribution maps** (separate from the shift): does the current driver map show which driver is significantly dominant, or just which has the highest weight? If the latter, add a confidence layer.

---

## PHASE 2 — Figures (publication quality)

All figures must be: 300 DPI minimum, fonts ≥ 8pt, colorbar labeled, no title-only annotation.

### Confirmed existing (may need polish)
- [~] **Driver maps** — `outputs/driver_maps/All_PFTs_SOS.png`, `All_PFTs_EOS.png`, per-PFT maps (check DPI and font sizes)
- [~] **Driver shift Δ maps** — `data/sensitivity_figures/AllPFT_combined_delta_panel.png`, per-PFT delta/dominance/trajectory (check quality)
- [~] **Attention window maps** — `data/attention_window/maps/All_PFTs_MultiPanel_WindowStart_SOS.png` (check if SHR is missing)
- [~] **Summary heatmap** — `data/sensitivity_figures/AllPFT_summary_heatmap.png`

### Figures still to build or finalize
- [ ] **Fig 1 (prediction skill overview)**: unified 6-panel figure — one 4-subplot scatter per PFT (overall / across-site / seasonal / anomaly R²) combined into a multi-panel. Currently only GRA and SHR exist as individual PNGs.
- [ ] **Fig 2 (anomaly skill bar chart or table)**: horizontal bar chart showing anomaly R² for all 6 PFTs side-by-side, with benchmark model values (4GST, PhenoFormer, ThermalTime) shown as reference lines. This is the "hardest number" figure.
- [ ] **Fig 3 (driver attribution maps)**: finalize `All_PFTs_SOS.png` and `All_PFTs_EOS.png` — confirm GRA/SHR included, decide whether to show RGB only or add dominant-driver overlay
- [ ] **Fig 4 (driver shift)**: select the best representation from the 83 sensitivity figures. Candidate: `AllPFT_combined_delta_panel.png` (Tier 1 Δ maps with stippling). Verify stippling is visible at publication size.
- [ ] **Fig 5 (attention timing)**: `All_PFTs_MultiPanel_WindowStart_SOS.png` — confirm all 6 PFTs present and legible
- [ ] **Extended Data / Supplementary**: per-PFT delta maps, zonal profiles, decadal trajectories (already generated), individual metrics figures

---

## PHASE 3 — Writing

Work in `paper/` directory. Write in order: Methods → Results → Abstract → Intro → Discussion (standard advice for Nature-style papers — results drive everything).

- [ ] **Methods** (write first — most mechanical):
  - Model architecture (TFT: hist_len=365, fut_len=30, state_size=160, 4 LSTM layers, 4 attention heads, quantile loss)
  - Data: LCSIF 1982–2021 daily 0.25°, ERA5 variables (tmin, tmax, radiation, precipitation, soil moisture, photoperiod)
  - Data splits: train <1992 + ≥2011; validation 1992–2001; test 2001–2011; 40-year test mode 1982–2022
  - PFT coverage: BDT (4 latitude bands), BET, NET, NDT, GRA, SHR
  - SOS/EOS detection method per PFT (slope-based generic; BET: latitude-DOY band matching)
  - Driver attribution: attention weights → max-attention 30-day window → median feature selection weights
  - Driver shift: epoch medians (1982–91 vs 2012–21), bootstrap 95% CIs (B=200), dominance coding
  - Transfer learning: pretrained on BDT, fine-tuned per PFT
  - Benchmarks: 4GST, PhenoFormer, ThermalTime (describe their methods briefly)

- [ ] **Results** (write second — the argument):
  - R1: Global prediction skill across PFTs (overall + seasonal)
  - R2: Interannual anomaly skill — R² = 0.69 for BDT, comparison to benchmarks (this is the novel number)
  - R3: Global driver attribution maps — spatial patterns of temperature / solar / water dominance
  - R4: 1982–2022 driver shift — water ↑, temperature ↓, spatial distribution and significance

- [ ] **Abstract** (150 words, write after Results):
  - Sentence 1: broad context (phenology matters for carbon cycle, climate feedback)
  - Sentence 2: problem (interannual anomaly skill is poor in existing models)
  - Sentence 3: approach (TFT on LCSIF + ERA5, 6 PFTs, 1982–2022)
  - Sentence 4: key result (R² = 0.69 anomaly, better than X)
  - Sentence 5: driver finding (water ↑, temp ↓ globally since 1982)
  - Sentence 6: implications (climate change is reorganizing phenological control)

- [ ] **Introduction** (~4 paragraphs, no headers for Nature):
  - P1: Phenology as a global-change indicator; carbon cycle + ecosystem timing stakes
  - P2: Limitations of existing models (process-based: parameter uncertainty; statistical: poor anomaly skill; ML: black-box attribution)
  - P3: What TFT offers: sequence modeling + interpretable attention; LCSIF enables 40-year global training
  - P4: What this paper does + roadmap (don't say "in this paper we show" — just state the findings)

- [ ] **Discussion** (~3 paragraphs):
  - D1: Interpret the anomaly R² = 0.69 — what does it mean mechanistically that TFT captures interannual variability where process models fail?
  - D2: Driver shift interpretation — consistent with drought stress increasing, VPD rise, Arctic amplification reducing temperature gradients? Compare to literature.
  - D3: Limitations + future work (BDT-centric training, transfer learning artifacts, LCSIF retrieval uncertainty, interpretability is attention not causality)

- [ ] **Cover letter** (1 page): lead with the anomaly skill result and driver shift; explain why Nature/NCC/NGeo specifically

---

## PHASE 4 — Submission preparation

- [ ] **Data availability statement**: LCSIF (cite source), ERA5 (Copernicus), predictions/attention data to be deposited on Zenodo or similar
- [ ] **Code availability statement**: this GitHub repo (paper/cleanup branch), DOI via Zenodo
- [ ] **Author contributions** (CRediT taxonomy)
- [ ] **Reviewer suggestions** (3–5 names): experts in global phenology, DL for Earth science, climate attribution
- [ ] **Conflicts of interest declaration**
- [ ] Final read for Nature style: no passive voice in abstract, SI methods vs main methods split, reference limit (~50 for Nature)

---

## Open questions (need decisions, not just execution)

1. **Is R² = 0.69 anomaly for BDT the right headline, or does it look weaker once other PFTs are computed?** Need all PFT numbers before deciding.
2. **Is the water↑/temperature↓ shift genuinely novel?** Some prior work (e.g., Yuan et al. 2019 Nature Climate Change, Zhu et al. global greening) touches related findings. Must check literature before committing to this as the primary claim.
3. **Is Nature flagship realistic in 1 week?** Nature requires full methods, SI, and typically 4–5 publication-quality figures. Nature Climate Change or Nature Geoscience have similar quality standards but more domain-appropriate scope and faster handling. Recommend targeting NCC or NGeo for the first submission; escalate to Nature only if the anomaly skill and driver shift are clearly stronger than anything in the literature folder.
4. **GRA and SHR in driver maps**: GRA and SHR are absent from `outputs/driver_maps/` individual files — confirm they appear in the All_PFTs combined figure before reporting them.
5. **Transfer learning artifact check**: BDT is the pretrained base; other PFTs are fine-tuned from BDT weights. If anomaly R² is much lower for non-BDT PFTs, reviewers will ask whether the transfer learning is limiting performance. Have a response ready.
