# Paper Tasks — Phenofusion → Nature Climate Change submission

**Target:** Nature Climate Change (see Phase 0 decision below)
**Deadline:** ~1 week
**Core claim:** TFT trained on LCSIF (1982–2021, 0.25°, daily) + ERA5 predicts global vegetation phenology (R² ≈ 0.99 overall; R² = 0.69 interannual anomalies for BDT — substantially exceeding process-based models); attention weights reveal a 1982–2022 driver shift: water availability increasingly dominant, temperature declining globally.

**Draft location:** `paper/nature_draft.tex`
**Ready figures:** `metrics.svg`, `pipeline.svg`, `attention_method.svg` (finalized); `driver_map.svg` (needs update)
**Sensitivity figures generated:** `data/sensitivity_figures/` (83 PNGs + `summary_table.csv`)

---

## Status legend
- [ ] TODO
- [~] In progress
- [x] Done

---

## PHASE 0 — Journal decision

**Decision: target Nature Climate Change (NCC), not Nature flagship.**

Reasoning from literature review:
- PhenoFormer (Garnot et al., Methods Ecol Evol 2025): country-scale Switzerland, 9 species, nRMSE metric on DOY — *not* global, *not* multi-PFT, *no* driver attribution. Our work is substantially broader.
- MIRANDA (Jiang et al., arXiv 2026): R² ≈ 0.49–0.50 on species-level DOY in Switzerland. Same limitation.
- Li et al. 2022 (Nat Comms): "Widespread increasing vegetation sensitivity to soil moisture" already showed the water↑ signal using NDVI/GPP. **Our driver shift finding is NOT completely novel — Li et al. got there first on the biological signal.** What IS novel: showing it specifically for phenological timing (SOS/EOS) via TFT attention attribution, across 6 PFTs, globally, with per-driver quantification.
- The anomaly R² = 0.69 story is the stronger novelty claim relative to the literature. Process models have near-zero anomaly skill (Richardson et al. 2012).
- **Nature flagship** would require the driver shift to be wholly new *and* paradigm-shifting. With Li et al. 2022 already published, that's a hard case. NCC is the right venue: it publishes global-scale climate×vegetation attribution with strong ML components.

**Revised framing for NCC:** Lead with the anomaly skill advance ("deep learning solves the interannual phenology problem that has defeated process models for decades"), then use driver attribution + temporal shift as the mechanistic story explaining *why* existing models fail (they assumed temperature primacy; water is increasingly dominant).

- [x] Journal target decided: Nature Climate Change

---

## PHASE 1 — Blocking: results must be complete before writing

### 1A. Confirm and complete the metrics table

These numbers are already in `nature_draft.tex` (Results §2.1) — they need to be verified from actual code output, not just trusted from the draft text. The draft currently says:
- BDT: R²_overall = 0.99, R²_anomaly = 0.69 ✓ (stated as confirmed)
- NDT: R²_overall = 0.99, R²_anomaly = 0.43 (stated in draft — **verify**)
- BET: R²_overall = 0.98, R²_anomaly = 0.45 (stated in draft — **verify**)
- NET: R²_overall = 0.99, R²_anomaly = 0.33 (stated in draft — **verify**)
- GRA: unknown — `outputs/metrics/metrics_GRA.png` exists, R² numbers not in draft
- SHR: unknown — `outputs/metrics/metrics_SHR.png` exists, R² numbers not in draft

Tasks:
- [ ] **Reproduce BDT anomaly R² = 0.69** by re-running `metrics_tft.py` (capture stdout, not just the figure). Confirm RMSE and BIAS too.
- [ ] **Get BET, NDT, NET anomaly R²** from SLURM runs — capture the printed `res` list. `metrics_tft.py` line 398 prints `[R²_overall, R²_across_site, R²_seasonal, R²_anomaly]`.
- [ ] **Get GRA and SHR anomaly R²** — read from the existing figure or re-run. These PFTs are completely absent from the draft results section.
- [ ] **Write `scripts/collect_metrics.py`** — a small script that loads each pred/data pair, calls `data_to_plot()` and captures the four R² values into a single CSV table. This replaces manual stdout reading.
- [ ] **Add GRA and SHR to the results text** — the current draft mentions only BDT/NDT/BET/NET. GRA and SHR are missing from every results paragraph.
- [ ] **Benchmark comparison** — retrieve anomaly R² or equivalent from 4GST, PhenoFormer, and ThermalTime for BDT. Data is at `/burg/glab/users/al4385/data/4GST/`, `phenoformer/`, `pyphenology_benchmark/`. Even one PFT (BDT) comparison is essential for the "advances over prior work" claim.

### 1B. Driver maps completeness

- [x] BDT Combined (4-band), BET, NDT, NET driver maps exist in `outputs/driver_maps/`
- [x] `All_PFTs_SOS.png` and `All_PFTs_EOS.png` exist — but need to **verify GRA/SHR are included** (open the PNGs)
- [ ] **GRA and SHR individual driver maps** — either confirm in `All_PFTs_*` or re-run `run_driver_maps.py` with GRA/SHR data from `ms7073/analysis/driversdata/oversampling/driver_maps_oversampling_20251204/`
- [ ] **Update `driver_map.svg`** — this is the figure that needs new results. It currently exists but needs to reflect the final GRA/SHR-inclusive all-PFTs composite.

### 1C. Driver shift quantification (for Results §2.4)

The `data/sensitivity_figures/summary_table.csv` already exists with per-PFT Δ values. Key numbers to extract for the paper:

- [x] Bootstrap Δ per driver computed for all 6 PFTs × 2 phases
- [x] `summary_table.csv` has: `mean_dtemp`, `mean_dsol`, `mean_dp`, `frac_sig_pos/neg` per driver

Specific tasks:
- [ ] **Compute global pixel-weighted mean Δ** per driver across all PFTs. From the CSV: BET SOS shows water↑ most strongly (mean_dp = +0.0068, frac_sig_pos_dp = 0.50); BDT/NET/NDT show temp↓ consistently. Need a weighted average.
- [ ] **Write the headline numbers** for the paper — e.g. "Across PFTs, the mean attribution weight of water availability increased by X ± Y percentage points (fraction of pixels with significant increase: Z%) while temperature declined by W ± V points."
- [ ] **Flag the heterogeneity honestly** — for BDT SOS, `frac_sig_neg_dtemp` = 0.51 and `frac_sig_pos_dtemp` = 0.41, meaning roughly equal fractions show significant positive and negative temp changes. The global mean is small (-0.0006). The shift is spatially heterogeneous — this needs to be stated in Results, not hidden.
- [ ] **GRA is anomalous** — GRA shows *opposite* pattern (`mean_dp` is *negative* = -0.0054 for SOS, water attribution *decreased*). This needs explanation in Discussion (grasslands may be limited by temperature at high latitudes more than water in temperate grasslands).

---

## PHASE 2 — Fix the draft (`nature_draft.tex`)

### Critical fixes (must do before submission)

- [ ] **Title**: "Blooming Insights" is too colloquial for NCC. Replace with a direct scientific claim. Candidate: *"Deep learning reveals a global shift from temperature- to water-controlled vegetation phenology over 1982–2022"*. Or lead with the skill advance: *"Transformer-based forecasting resolves interannual phenological variability and reveals shifting climate drivers across global biomes"*.

- [ ] **Abstract**: Currently buries the key finding. Rewrite to:
  1. Context (1 sentence): phenology is a climate-change indicator; interannual variability drives ecosystem carbon uncertainty
  2. Problem gap (1 sentence): process-based models fail at interannual anomalies; attribution methods lack global scale
  3. Approach (1 sentence): TFT on 40 years of LCSIF + ERA5, 6 PFTs, global 0.25°
  4. Key result 1 (1 sentence): R² = 0.69 for interannual anomalies (BDT), substantially exceeding process models
  5. Key result 2 (1 sentence): attention-based attribution reveals water ↑, temperature ↓ as dominant trends 1982–2022
  6. Implication (1 sentence): phenological forecasts under climate change need to account for shifting driver importance, not fixed temperature-centric parameterizations

  Current abstract says "R² ~ 1" — this is misleading as the headline. R² ~ 1 is the *overall* fit (trivially high for a model that can learn climatology). The *anomaly* R² = 0.69 is the scientifically meaningful number.

- [ ] **PFT naming inconsistency**: Discussion uses "DBT", "EBT", "DNT", "ENT" — should be BDT, BET, NDT, NET throughout. Find and replace.

- [ ] **Training split**: Methods says "50% training, 25% validation, 25% testing" with dates "1982-1992 and 2011-2021" for training, "1992-2002" for validation, "2002-2010" for testing. Fix: the training period is 1982–1992 *plus* 2011–2021 (~20 years total), validation 1992–2001, test 2001–2011. The "40-year test mode" (1982–2022, all data) is what was used for driver analysis. Clarify this clearly — it is non-standard and reviewers will ask about it.

- [ ] **EOS section is empty** (§Senescence is blank after the header). Write it — the driver maps figure (`driver_map.svg`) includes EOS, so the patterns just need to be described.

- [ ] **Legacy effects section** is incomplete ("explain typical time scales in more details"). Fill in: what time scale? Which PFTs? The attention window maps (`data/attention_window/maps/`) give this directly.

- [ ] **Long-term shifts section** (§2.4): currently qualitative only. Add the headline Δ numbers from `summary_table.csv`. Must quantify: by how much did water attribution change? What fraction of pixels showed significant shifts?

- [ ] **Add GRA and SHR everywhere**: results tables/figures, driver map descriptions, sensitivity shift discussion.

- [ ] **Benchmark comparison paragraph**: add one paragraph comparing anomaly R² to process-based models (cite Richardson et al. 2012, Peano et al. 2021 for process model skill) and to PhenoFormer/statistical approaches. Key point: PhenoFormer operates on species-level DOY in Switzerland — different scale and metric, so comparison is indirect but instructive.

- [ ] **Cite Li et al. 2022 Nat Comms** in the discussion of driver shifts — this paper already showed "widespread increasing vegetation sensitivity to soil moisture." Distinguish your finding: you show this specifically for phenological timing via ML attention, not just for photosynthesis/NDVI.

- [ ] **Cite "Drought hinders spring phenology" (NCC 2025)** — currently in references.bib (`noauthor_drought_2025`) but not cited in the draft. Directly relevant to memory effects section.

- [ ] **Template**: Current `\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}` is Springer Nature format. Nature Climate Change uses its own submission system — for writing purposes this is fine, but the final submission requires NCC's Word template or the standard NCC LaTeX class. Low priority for now; switch at submission stage.

### Lower priority fixes

- [ ] The `\` on line 304 (before `\bibliography`) is a stray command — remove it.
- [ ] Figure references use `Figure~\ref{fig:metrics}` in some places and `Figure \ref{fig:drivers}` (with space before \ref) in others — standardize.
- [ ] `\subsubsection` is used throughout Results and Discussion — NCC format uses fewer heading levels. Consider flattening to `\subsection` only.
- [ ] "Figure Y1–Y4" (line 236) is a placeholder — update with actual figure/supplementary numbers once finalized.
- [ ] Driver maps caption mentions "Hue indicates the dominant driver, while saturation reflects driver strength" — verify this is accurate for the RGB ternary colormap being used.

---

## PHASE 3 — Figures (finalization)

### Ready to use as-is
- [x] `pipeline.svg` — modeling pipeline overview (Fig 1 or Extended Data)
- [x] `attention_method.svg` — attention interpretability illustration
- [x] `metrics.svg` — prediction skill scatter plots (need to confirm GRA/SHR are included)

### Needs update
- [ ] **`driver_map.svg`**: update with final all-PFTs maps including GRA/SHR. Decision needed: use the all-PFTs RGB ternary composite (`All_PFTs_SOS.png`, `All_PFTs_EOS.png`) or the per-PFT 6-panel layout. NCC typically has 4–6 main figures; this is likely one of them.

### Needs to be built (main figures for NCC)
- [ ] **Fig for driver shift** (Tier 1 Δ maps): best candidate is `data/sensitivity_figures/AllPFT_combined_delta_panel.png`. Need to check: are stippling dots visible at NCC print size (87mm single column, 180mm double column)? Reproduce at 300 DPI in vector if needed.
- [ ] **Fig for anomaly skill** (the key performance figure): build a clean horizontal bar chart or grouped bar chart with R²_anomaly for all 6 PFTs, with benchmark comparisons shown as reference lines or grouped bars. This is the figure that directly supports the core claim.
- [ ] **Summary heatmap** (`AllPFT_summary_heatmap.png`): candidate for a figure showing overall driver patterns across PFTs × phases. Check quality.
- [ ] **Attention window / legacy effects map**: `All_PFTs_MultiPanel_WindowStart_SOS.png` — check if SHR is present and figure is NCC-quality.

---

## PHASE 4 — Submission preparation (after draft is complete)

- [ ] **Data availability**: LCSIF (Fang et al. 2024, already cited), ERA5 (C3S, already cited), ESA PFT v2.0.8 (Harper et al. 2023, already cited). Attention weights and SOS/EOS attribution CSVs → deposit on Zenodo.
- [ ] **Code availability**: this GitHub repo (`paper/cleanup` branch) → get DOI via Zenodo.
- [ ] **Author contributions** (CRediT): Aya Lahlou, Maria Fernanda Sampaio Ferreira, Linnia Hawkins, Pierre Gentine.
- [ ] **Reviewer suggestions** (3–5 names): target experts in global phenology remote sensing, interpretable ML for Earth sciences, climate-vegetation attribution (e.g., someone who has published on SIF-phenology, someone from the LSM phenology intercomparison community).
- [ ] **Cover letter**: lead with anomaly R² = 0.69 vs near-zero for process models; second paragraph on driver shift and its climate change implications; third paragraph on why NCC specifically.
- [ ] **Competing interests** declaration.

---

## Open questions requiring decisions

1. **Are the anomaly R² numbers in the draft actually correct?** (BET = 0.45, NDT = 0.43, NET = 0.33) — These need to be reproduced, not just trusted from draft text. If GRA/SHR anomaly R² is low (say < 0.2), does that change the headline claim?

2. **How to handle GRA's opposite driver shift signal?** `summary_table.csv` shows GRA SOS has `mean_dp` = -0.0054 (water attribution *decreased*). This needs to either be explained ecologically (grasslands in temperate zones are more temperature-limited; the water signal is biome-specific) or qualified in the abstract.

3. **Is the driver shift statistically robust enough to be a headline finding?** BDT SOS has roughly equal fractions of pixels with significant positive vs negative temp changes (~41% up, ~51% down). The *mean* Δtemp = -0.0006 is tiny. BET shows a much cleaner signal (frac_sig_neg_dtemp = 0.41, frac_sig_pos_dp = 0.50, mean_dp = +0.0068). The claim "temperature declining, water increasing" is true globally on average but with large regional heterogeneity. The abstract should say "globally averaged" or present the BET result as a case study rather than a universal claim.

4. **PFT abbreviations in the paper**: the text uses both "broadleaf deciduous trees (BDT)" and "DBT" (in discussion). Standardize to BDT/BET/NDT/NET/GRA/SHR throughout.

5. **Transfer learning caveat**: all non-BDT PFTs are fine-tuned from BDT weights. If anomaly R² is lower for non-BDT PFTs, reviewers will ask if transfer learning is limiting performance. Prepare a response: the anomaly R² for NET (0.33) could reflect both ecological complexity and transfer learning artifacts. Either run an ablation (train NET from scratch) or explicitly acknowledge this as a limitation.
