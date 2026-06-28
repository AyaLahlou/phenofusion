# Paper Tasks — Phenofusion → Nature Climate Change submission

**Target:** Nature Climate Change (see Phase 0 decision below)
**Deadline:** ~1 week
**Core claim:** TFT trained on LCSIF (1982–2021, 0.25°, daily) + ERA5 predicts global vegetation phenology (R² ≈ 0.99 overall; R² = 0.69 interannual anomalies for BDT — substantially exceeding process-based models); attention weights reveal a 1982–2022 driver shift: water availability increasingly dominant, temperature and solar radiation declining globally.

**Draft location:** `paper/nature_draft.tex`
**Ready figures:** `metrics.svg`, `pipeline.svg`, `attention_method.svg` (finalized); `driver_map.svg` (needs update)
**Sensitivity figures generated:** `data/sensitivity_figures/` (83 PNGs + `summary_table.csv`)
**Legacy depth maps:** `data/attention_window/maps/` (per-PFT and all-PFT combined)

---

## Status legend
- [ ] TODO
- [~] In progress
- [x] Done

---

## PHASE 0 — Journal decision

**Decision: target Nature Climate Change (NCC), not Nature flagship.**

Reasoning from literature review:
- PhenoFormer (Garnot et al., Methods Ecol Evol 2025): country-scale Switzerland, 9 species, nRMSE metric on DOY — *not* global, *not* multi-PFT, *no* driver attribution.
- MIRANDA (Jiang et al., arXiv 2026): R² ≈ 0.49–0.50 on species-level DOY in Switzerland. Same limitation.
- Li et al. 2022 (Nat Comms): "Widespread increasing vegetation sensitivity to soil moisture" already showed the water↑ signal using NDVI/GPP. **Our driver shift finding is NOT completely novel — Li et al. got there first on the biological signal.** What IS novel: showing it specifically for phenological timing (SOS/EOS) via TFT attention attribution, across 6 PFTs, globally, with per-driver quantification.
- **Revised framing for NCC:** Lead with the anomaly skill advance ("deep learning solves the interannual phenology problem that has defeated process models for decades"), then use driver attribution + temporal shift as the mechanistic story explaining *why* existing models fail.

- [x] Journal target decided: Nature Climate Change

---

## PHASE 1 — Blocking: results must be complete before writing

### 1A. Confirm and complete the metrics table

Current status in `nature_draft.tex` (from figure images, not re-run):
- BDT: R²_overall = 0.99, R²_anomaly = 0.69
- NDT: R²_overall = 0.99, R²_anomaly = 0.43
- BET: R²_overall = 0.98, R²_anomaly = 0.45
- NET: R²_overall = 0.99, R²_anomaly = 0.33
- GRA: R²_overall = 0.99, R²_anomaly = 0.24 ← read from `outputs/metrics/metrics_GRA.png`
- SHR: R²_overall = 0.98, R²_anomaly = 0.67 ← read from `outputs/metrics/metrics_SHR.png`

- [ ] **Reproduce all anomaly R² by re-running `metrics_tft.py`** — capture stdout `res` list for each PFT. The image-read numbers are adequate for drafting but need programmatic confirmation before submission. Consider `scripts/collect_metrics.py`.
- [x] **GRA and SHR anomaly R² obtained** (from figure images) and written into draft
- [ ] **Benchmark comparison** — retrieve anomaly R² or equivalent from 4GST, PhenoFormer, ThermalTime for BDT. Data at `/burg/glab/users/al4385/data/4GST/`, `phenoformer/`, `pyphenology_benchmark/`. At minimum one PFT (BDT) for the "advances over prior work" claim.

### 1B. Driver maps completeness

- [x] BDT Combined (4-band), BET, NDT, NET driver maps exist in `outputs/driver_maps/`
- [x] `All_PFTs_SOS.png` and `All_PFTs_EOS.png` exist in `outputs/driver_maps/`
- [ ] **Confirm GRA and SHR are in `All_PFTs_*.png`** — visually inspect the PNGs to verify GRA/SHR pixels are included (the text now describes them; the figure must match)
- [ ] **Update `driver_map.svg`** — current SVG predates GRA/SHR. Replace with final all-PFTs composite including all 6 PFTs for both SOS and EOS panels.

### 1C. Driver shift quantification

- [x] Bootstrap Δ per driver computed for all 6 PFTs × 2 phases (sensitivity pipeline done)
- [x] Headline numbers written into §Long-term Shifts using `data/sensitivity_figures/AllPFT_summary_heatmap.png`
- [x] Spatial heterogeneity of temperature signal acknowledged in draft (BDT SOS: 51% sig neg, 41% sig pos)
- [x] GRA anomalous pattern explained in draft (water attribution decreased at SOS)
- [ ] **Verify NDT EOS Δwater = +0.008** — the heatmap pixel was small; confirm against `summary_table.csv` row for NDT EOS

---

## PHASE 2 — Fix the draft (`nature_draft.tex`)

### Critical fixes — DONE this session

- [x] **EOS / Senescence section** — was empty; now fully written (3 paragraphs: boreal photoperiod dominance, water less prominent than SOS, PFT-specific NDT/BET/GRA/SHR patterns)
- [x] **Legacy effects section** — was vague stub; now quantitative (BET/NDT median window_start≈24–27 → 12 months prior; NET SOS ≈81 → 9 months; BDT/GRA/SHR = 335 → current season). Added `fig:legacy` figure.
- [x] **Long-term shifts section** — was qualitative only; now PFT-by-PFT Δ values with significance fractions. Added `fig:sensitivity` figure.
- [x] **GRA and SHR added everywhere** in Forecasting Performance, Senescence, Legacy Effects, Long-term Shifts, and Discussion
- [x] **PFT naming inconsistency fixed** — DBT/EBT/DNT/ENT → BDT/BET/NDT/NET throughout Discussion
- [x] **Stray `\`** before `\bibliography` removed

### Critical fixes — still TODO

- [ ] **Title**: "Blooming Insights" is too colloquial for NCC. Candidates:
  - *"Deep learning reveals a global shift from temperature- to water-controlled vegetation phenology over 1982–2022"*
  - *"Transformer-based forecasting resolves interannual phenological variability and reveals shifting climate drivers across global biomes"*

- [ ] **Abstract**: rewrite 6-sentence structure (context → gap → approach → result 1: anomaly R²=0.69 → result 2: water↑ shift → implication). Current version buries R²=0.69 and leads with "R²~1" which is scientifically unimpressive. GRA/SHR need to be mentioned or scope qualified.

- [ ] **Training split clarification** in Methods: currently says "50%/25%/25%" with inconsistent dates. Correct: train = 1982–1992 + 2011–2021 (~20 yrs), validation = 1992–2001, test = 2001–2011. Driver analysis used the full 40-year test mode (1982–2022, all data as test). The non-standard split needs explicit explanation — reviewers will ask.

- [ ] **Benchmark comparison paragraph** — add to Discussion §Improved Accuracy: compare anomaly R² to PhenoFormer (different scale/metric, indirect) and to process-based model skill (near-zero per Richardson et al. 2012, Peano et al. 2021). This is the paragraph that earns the "significant advancement" claim.

- [ ] **Cite Li et al. 2022 Nat Comms** in Discussion §Climate Change Impacts — distinguish our finding (phenological timing specifically, via ML attention) from their photosynthesis/NDVI result.

- [ ] **Cite "Drought hinders spring phenology" (NCC 2025)** (`noauthor_drought_2025` in references.bib) — cite in Legacy Effects section (directly relevant to the BET/NDT annual memory finding).

### Lower priority fixes

- [ ] Figure references: standardize spacing — `Figure~\ref{...}` everywhere (some places have `Figure \ref{...}` or `Figure ~\ref {...}`)
- [ ] "Figure Y1–Y4" placeholder (line ~236 in original) — update with actual supplementary figure numbers once layout is finalized
- [ ] Driver maps caption: "Hue indicates dominant driver, saturation reflects driver strength" — verify this description is accurate for the RGB ternary colormap actually used
- [ ] `fig:legacy` and `fig:sensitivity` use relative PNG paths (`../data/...`) — confirm these resolve correctly when compiling from `paper/` directory
- [ ] Consider flattening `\subsubsection` to `\subsection` for NCC format (NCC uses fewer heading levels)
- [ ] Template: `sn-mathphys-num` is Springer Nature, not NCC format — switch at submission stage (low priority now)

---

## PHASE 3 — Figures (finalization)

### Ready to use as-is
- [x] `pipeline.svg` — modeling pipeline overview
- [x] `attention_method.svg` — attention interpretability illustration
- [ ] `metrics.svg` — confirm GRA and SHR panels are included in the SVG (draft now references all 6 PFTs; figure must match)

### Needs update
- [ ] **`driver_map.svg`**: update with GRA/SHR-inclusive all-PFTs composite. Options: (a) use `All_PFTs_SOS.png` + `All_PFTs_EOS.png` as-is and convert to SVG container, or (b) re-run `run_driver_maps.py` with all 6 PFTs and export SVG. Decision needed.

### Needs to be built (new main figures)
- [ ] **Anomaly skill bar chart** — horizontal grouped bar chart: R²_anomaly for all 6 PFTs, with process-model benchmark as reference line/bar. This is the primary figure supporting the core claim. (Python, ~20 lines)
- [ ] **Driver shift Δ panel** (`fig:sensitivity`): `data/sensitivity_figures/AllPFT_combined_delta_panel.png` is drafted as a placeholder PNG. For NCC publication: check stippling visibility at 87mm column width; re-export at 300 DPI if needed.
- [ ] **Legacy depth map** (`fig:legacy`): `All_PFTs_LegacyDepth_SOS.png` is strong — verify DPI and add EOS companion panel, or reference it in supplementary.
- [ ] **Summary heatmap** (`data/sensitivity_figures/AllPFT_summary_heatmap.png`): good candidate for supplementary Extended Data — shows full PFT×phase×driver matrix.

---

## PHASE 4 — Submission preparation (after draft is complete)

- [ ] **Data availability**: LCSIF (Fang et al. 2024, cited), ERA5 (C3S, cited), ESA PFT v2.0.8 (Harper et al. 2023, cited). Attention weights + SOS/EOS attribution CSVs → deposit on Zenodo.
- [ ] **Code availability**: GitHub repo (`paper/cleanup` branch) → Zenodo DOI.
- [ ] **Author contributions** (CRediT): Aya Lahlou, Maria Fernanda Sampaio Ferreira, Linnia Hawkins, Pierre Gentine.
- [ ] **Reviewer suggestions** (3–5 names): global phenology remote sensing, interpretable ML for Earth sciences, climate-vegetation attribution.
- [ ] **Cover letter**: lead with anomaly R²=0.69 vs near-zero for process models; driver shift + implications; why NCC.
- [ ] **Competing interests** declaration.

---

## Open questions

1. **Are BET/NDT/NET anomaly R² in the draft actually correct?** (0.45 / 0.43 / 0.33) — From draft text, not re-run. GRA/SHR were read from figure images. All six PFT values need programmatic confirmation before submission.

2. **Does SHR anomaly R²=0.67 hold up on re-run?** If confirmed, it's a key finding worth highlighting — shrubs outperforming most tree PFTs on interannual variability is ecologically meaningful and should be given a sentence in the abstract.

3. **Transfer learning caveat**: all non-BDT PFTs are fine-tuned from BDT weights. Lower anomaly R² for NET/GRA may partly reflect transfer learning artifacts. Prepare a response: either run an ablation (train NET from scratch) or acknowledge this explicitly as a limitation.

4. **Are GRA and SHR present in `All_PFTs_SOS.png` and `All_PFTs_EOS.png`?** If not, those figures cannot be used as-is and `driver_map.svg` update is blocked pending a re-run of `run_driver_maps.py`.

5. **Driver shift framing in abstract**: the water↑ signal is cleanest for BET (50% pixels sig positive, 0% sig negative at SOS). For BDT it's noisier (54% sig pos but 37% sig neg). The abstract should say "in most PFTs" or lead with BET as the clearest example rather than claiming a universal global trend.
