# Review item 6: ground validation of the observed (LCSIF) sensitivity, and scoring of the benchmark models

**Question (review §6 item 6).** (a) Does the LCSIF-derived SOS/EOS, and the temperature sensitivity (ST, d/°C) computed from it, agree with ground phenology (PhenoCam, USA-NPN, PEP725)? How much would a 15-day, 0.25° proxy attenuate ST? (b) Actually score 4GST, PhenoFormer and ThermalTime against the same targets as the TFT.

All compute was CPU (no GPU used). Job IDs are in `outputs/review/item6/jobs.tsv`; scripts in `scripts/review/item6/`, SLURM wrappers in `bashscripts/review/item6/`, small tables in `data/review/item6/`, big files under `/burg-archive/glab/users/al4385/{data,analysis}/review/item6/` (about 0.5 GB in total).

## 1. Ground data

| Source | Status | What I used |
|---|---|---|
| PhenoCam v3 (public API) | downloaded (748 ROI transition files, 2000–2025) | `gcc_90`, `transition_50` (50 % of GCC amplitude), rising = SOS, falling = EOS; one ROI per site and vegetation type; the largest-amplitude cycle per calendar year. Veg type to PFT: DB→BDT, EN→NET, DN→NDT, GR→GRA, SH→SHR, EB→BET |
| USA-NPN (public API) | downloaded (site-level phenometrics 2009–2021) | SOS = breaking leaf buds / initial growth / emerging needles; EOS = colored leaves / needles (falling leaves only if no colour record); first-yes with a prior "no" within 14 d; median over species per site-year; species functional type to PFT |
| PEP725 | **not available. Gap.** | No local copy exists under `/burg-archive/glab/users/al4385/`, the repo `data/`, or the home tree (searched `*pep*`, `*PEP725*`). Access needs registration at pep725.eu and I did not register. The matching code is source-agnostic (`ground_build.py` / `compare_ground.py`), so a PEP725 table with columns `site,lat,lon,pft,phase,year,doy` can be appended to `ground_siteyears.csv` and the pipeline rerun unchanged. PEP725 is the only source with a long (1982+) European overlap with LCSIF, so this gap is what limits the interannual validation. |

Matching (`ground_build.py`): the nearest 0.25° grid point, in the pixel table of the site's own PFT dataset (`pft_match=True`, primary) or of any PFT dataset at that point (`pft_match=False`, sensitivity). Result: `data/review/item6/ground_siteyears.csv`, `ground_pixels.csv`. Sites with a pixel of their own PFT: PhenoCam BDT 175, NET 44, BET 7, SHR 10, GRA 1, NDT 0; NPN BDT 2,746 sites, NET 36, SHR 28, others about 0. **The validation is therefore essentially a BDT result (PhenoCam and NPN) with thin NET/SHR evidence; GRA, NDT and BET cannot be validated with these two networks.** Years overlap only 2008–2021 (14 y max), which gives at most 14 site-years per site.

## 2. LCSIF dates at the ground pixels (`lcsif_pixels.py`)

Three definitions, all per season-year:
- `smooth`: obs_st.py (31-d×2 smoothing, half-amplitude crossing). This is the definition of the item-2 observed ST.
- `node`: the same crossing rule on the native semi-monthly composites with no daily smoothing. The 1st/15th-of-month days are the knots of the interpolated daily record (verified in `find_nodes.py`/`find_nodes2.py`: the second differences are constant between knots, with kinks at DOY ≈ 5, 11, 21, 26, ...; item 2 found the same).
- `ev2`: the FINDINGS §2 event rule (30-d strictly monotonic window inside the latitude DOY band; date = window centre of the first qualifying run).
Input: `CSIFMETEO_smfix` where it exists (CSIF and temperature are bit-identical to the raw files).

## 3. Results: LCSIF vs ground (`compare_ground.py`, `data/review/item6/ground_vs_lcsif.csv`)

Own-PFT pixels, NH (≥25°N), `smooth` definition. Bias = median(LCSIF − ground) in days; r_within = interannual correlation after removing each site's mean (sites with ≥5 matched years); ST = pooled within-site OLS slope of the date on the mean pixel T over the 60 days ending at the site's mean ground date (the same window and the same ERA5 T for both date series). The ratio is ST_LCSIF/ST_ground with a 95 % site-bootstrap interval.

| Source | PFT | Phase | site-years | sites (≥5 y) | bias (d) | MAE (d) | r_within | ST ground | ST LCSIF | ratio [95 % CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| PhenoCam | BDT | SOS | 519 | 48 | +1 | 6.9 | **0.74** | −2.34 | −2.10 | **0.90 [0.79, 1.02]** |
| PhenoCam | BDT | EOS | 559 | 56 | −5 | 13.3 | 0.11 | +2.68 | +0.55 | 0.21 [0.09, 0.37] |
| PhenoCam | NET | SOS | 106 | 6 | +17 | 29 | 0.69 | −1.56 | −0.46 | 0.29 [−0.18, 0.91] |
| PhenoCam | NET | EOS | 106 | 7 | −16 | 22 | 0.10 | +0.29 | +0.58 | not meaningful |
| NPN | BDT | SOS | 3361 | 184 | +20 | 30.5 | 0.35 | −3.17 | −1.38 | 0.43 [0.37, 0.50] |
| NPN | BDT | EOS | 2758 | 162 | +10 | 38 | 0.08 | +0.41 | +0.33 | not meaningful |
| NPN | NET, SHR, GRA, BET | | ≤60 | ≤3 | | | | | | too few to estimate |

Other points:
- `node` and `smooth` agree for SOS (PhenoCam BDT: bias +2 vs +1, r_within 0.72 vs 0.74, ST −2.19 vs −2.10). The 15-day node resolution does not change the result.
- `ev2` is not a date estimator. Its interannual sd is 0.1–0.5 d against 5–13 d on the ground side, because the date is the centre of a 30-d window anchored to a fixed DOY band. Its ST is about 0 by construction. This agrees with the review's remark that the event definition makes the "nowcast" structure nearly sensitivity-free. I recommend not using `ev2` dates for any timing claim.
- Sensitivity run with sites matched to a pixel of another PFT dataset (`pft_match=False`, not primary): the ratio is 0.8–0.9 for grass and for NPN needleleaf and broadleaf-evergreen SOS, but 0.2–0.5 for PhenoCam NDT, NET and SHR SOS and 0.03–0.5 for EOS (`ground_vs_lcsif.csv`). These are mixed-cover pixels with few sites and I do not interpret them.
- Spring ST at the PhenoCam BDT sites is −2.1 (LCSIF) versus the item-2 PFT-wide pixel median of −1.22 (ST60, 78,749 pixels). The sites are a self-selected sample of well-instrumented, mostly eastern-North-American deciduous forests, so this is a representativeness difference between the two samples, not evidence that the global median is biased.

## 4. Expected attenuation of ST in a semi-monthly, 0.25° proxy (`sim_attenuation.py`, `sim_attenuation.csv`)

Synthetic pixels (40 y, true ST = −3 d/°C, residual date sd 5 d, T sd 1.5 °C) are passed through the actual chain (semi-monthly composites, cubic spline, 31-d×2 smoothing, half-amplitude crossing). 36 scenarios × 300 replicates varied transition width (3, 8, 20 d), composite noise (0, 5, 15 % of amplitude), subpixel date spread (0 or 12 d) and amplitude covarying with T.
- **Median recovered/true ratio is 0.92–1.04 in every scenario.** Neither the 15-day sampling nor the smoothing nor subpixel mixing attenuates a date-shift response. The half-amplitude crossing shifts one-for-one with the curve.
- Noise does not bias the slope; it widens it: the s.d. of the single-pixel 40-y ST estimate is 0.5 d/°C (no noise) to 1.1–1.6 d/°C (15 % noise, slow green-up). The 5–95 % range of the ratio is 0.2–1.8 in the worst case. This matches the item-2 pixel MDE80 (1.1–2.4 d/°C).
- The date-error correlation falls to 0.5–0.8 at 15 % noise, which bounds the interannual r one can expect between any LCSIF-derived date and truth. PhenoCam BDT SOS (r_within 0.74) is consistent with the low-noise end.
- Caveat: the simulation covers only the sampling/smoothing chain. It cannot capture the physical or definitional differences between canopy greenness, SIF and leaf-out. The empirical ratio shows this: 0.90 against PhenoCam BDT (where GCC and SIF track the same canopy flush) versus 0.43 against NPN BDT, whose first-leaf event is about 20 d earlier than the half-amplitude crossing and refers to individual understory-inclusive trees. The NPN number is a phenophase-definition gap and should not be read as proxy attenuation.

**Net statement.** For deciduous broadleaf SOS, LCSIF dates reproduce PhenoCam spring-green-up dates with about 1 d bias, 7 d MAE, interannual r = 0.74 and an ST within about 10 % of the ground value (interval 0.79–1.02). The temporal smoothing of the product is not what limits ST. For EOS, and for evergreen, shrub and grass types, there is no usable validation here (r_within ≈ 0.1 for EOS, n ≤ 7 sites for the others).

## 5. Benchmarks (`score_windows.py`, `pf_perturb.py`, `tt_summary.py`)

### 5a. What the stored benchmark outputs are (format issues)
- All predictions are `(N, 30, 3)` arrays with **no ids**; they align with the 2024 `TFT_30` test split (2002–2010) only by row order. I checked N and the skill (`NDT/NET/BDT` N matches, R² .79–.99). **BET TFT2025 does not align (149,142 rows vs 148,466)** and is dropped; the BET benchmark rows are unaffected.
- The "4GST" file is **not the 4GST algorithm**. `code/benchmark/FourGST_forecaster.py` is an OLS line through the last 30 days of SIF, extrapolated 30 days (I recomputed it; max difference to the stored file 9e-7). It uses no weather, so its ST is 0 identically.
- "PhenoFormer" is **PhenoFormerSeq**: a 4-layer, 160-d transformer encoder from the history of all 7 inputs (including past SIF) to the 30-d forecast, trained with MSE, a replicated median in all three quantile slots, no future weather, and positional encoding by window position instead of DOY. It is the published PhenoFormer only in name.
- **"ThermalTime"** (`pyphenology_benchmark/BET_ThermalTime_predictions.pkl`, BET only; the other pyPhenology models timed out in Aug 2025, `pyPhenology_Benchmark.out`). It fits ThermalTime to the *mean scaled CSIF of the target window, treated as if it were a DOY*, then multiplies by a mean shape template. That is not a phenology date model. Its R² is −1.31 (Overall; `pyphenology_benchmark/metrics`), and it is a constant-shape forecast, so it cannot be scored as a timing model. I report its window errors only to document that. No pyPhenology weights were saved, so its ST cannot be computed. `pyphenology_formatted/` holds only inputs.
- An LSTM benchmark exists in `~/code/benchmark/results/lstm` but is not one of the three requested models; I included it in the window scores as an extra reference (its categorical embeddings are not saved, so I did not perturb it).
- No GRA or SHR predictions exist for any benchmark. I did not run GRA/SHR window scores because there is nothing to score; the earlier assumption that they were complete was wrong.

### 5b. Window-level skill on identical targets
Events: the FINDINGS §2 rule applied to the TFT_30 test split (strictly monotonic 30-d observed window in the latitude DOY band). Timing = half-amplitude crossing day of the 30-d window (`threshold = (y₀+y₂₉)/2` in raw SIF units); the model is censored to day 30 when it never crosses. TFT2025 = `pred_{PFT}_feb202025.pkl` (BDT `pred_merged_BDT_feb172025.pkl`).

Median absolute timing error (days) / bias (days) / fraction with no crossing:

| PFT, phase | TFT2025 | PhenoFormerSeq | LSTM | 4GST (OLS extrapolation) | persistence |
|---|---|---|---|---|---|
| BDT SOS (n=40,403) | 1.4 / +0.7 / 5 % | 2.5 / −0.3 / 4 % | 4.4 / +0.9 / 13 % | 11.5 / +9.2 / 61 % | 14.6 / – / 100 % |
| BDT EOS (n=40,210) | 0.9 / +0.3 / 2 % | 1.5 / +0.8 / 1 % | 2.8 / 0.0 / 5 % | 9.5 / +7.1 / 34 % | 15.6 / – / 100 % |
| NET SOS (n=6,384) | 3.8 / −1.8 | 2.5 / +0.8 | 3.2 / −0.1 | 12.2 / +10.1 | |
| NET EOS (n=6,422) | 2.5 / +1.0 | 1.5 / +0.2 | 2.1 / −0.5 | 8.1 / +7.4 | |
| NDT SOS (n=1,868) | 2.9 / −2.3 | 1.8 / −0.6 | 2.3 / +1.8 | 14.8 / +14.4 | |
| NDT EOS (n=2,601) | 2.1 / +1.5 | 1.6 / +1.0 | 2.6 / −2.2 | 12.1 / +10.8 | |
| BET SOS (n=8,201) | not aligned | 3.7 / +2.0 | 5.3 / +0.6 | 10.7 / +7.7 | ThermalTime: 11.1 / −10.8 |
| BET EOS (n=7,223) | not aligned | 3.0 / +0.5 | 3.9 / −0.3 | 9.8 / +5.1 | ThermalTime: 14.6 / −7.8 |

Full table: `data/review/item6/window_scores_{BDT,NET,NDT,BET}.csv` (also trajectory R² and pooled r). Per-event parquet: `analysis/review/item6/windows/`.

Reading it:
- PhenoFormerSeq is **as good as or better than the TFT2025 checkpoints** on timing of these events for NET, NDT (and slightly worse for BDT). The TFT's advantages in the review's framing (future weather, probabilistic output) do not show up in transition-day accuracy at 30-d horizon. All three learned models are within about 1–4 d median error; the persistence and linear-extrapolation baselines fail by 10–16 d.
- `r_pooled` (pooled across events) is 0.2–0.6 for the learned models, and mainly spatial. **Interannual r could not be computed at the window level**: one window per location in the DOY band (`n_loc_ge4_years` = 0 except 2 BDT locations), because the TFT_30 test windows are about 395 d apart. The only interannual-skill numbers available are the long-lead values already in `compare_obs.csv` (r 0.06–0.5, pre-item-2 inputs).
- These are 30-d forecasts from a state that already contains the transition (item 1/3 caveats); they say nothing about long-lead skill.

### 5c. Sensitivity (ST) of the benchmarks
- **4GST**: ST = 0 exactly (no weather input).
- **PhenoFormerSeq** (`pf_perturb.py`; 3,000 random event windows per PFT and phase; unperturbed forecast reproduces the stored predictions to ≤ 5e-6): adding +1 °C to tmin and tmax over the last 60 history days shifts the SOS crossing by (median, d/°C) BDT −0.15 (mean −0.23), NET −0.42, NDT −0.30, BET −0.24. Whole-history +1 °C: BDT 0.00, NET −0.18, NDT −0.10, BET +0.06. The EOS shifts are +0.06 to +0.27 d/°C (observed EOS ST60 is about −0.4 to +0.3, item 2). The magnitude is tenths of a day per °C against an observed SOS ST of −1.2 to −1.7 (LCSIF, item 2) and −2.3 (PhenoCam BDT). The sign is right for SOS and the amplitude is 5–15 times too small. The existing TFT history-warming battery is of the same order or smaller for the 30-d checkpoints (|dt| ≤ 1e-3 d for SOS; `data/ideas/idea1/battery_summary.csv`), i.e. the benchmark is slightly more responsive than the TFT, but neither reaches the observed sensitivity. Table: `pf_perturb_{BDT,NET,NDT,BET}.csv`.
- **ThermalTime**: the benchmark file above is unusable for ST. As the nearest honest substitute I scored the minimal thermal-time model already built as the positive control in `obs_st.py` (GDD base 5 °C, a single median GDD requirement per pixel, SOS only, `thermaltime_obs_st.csv`). It is not a fitted pyPhenology model. Its interannual agreement with the CSIF-derived SOS is r = 0.57 (GRA), 0.52 (NDT), 0.55 (NET), 0.29 (BDT), and −0.10 / −0.09 (BET / SHR, the water-limited PFTs); its true response is −3.0 to −5.9 d/°C, i.e. 2–3× the observed value (−1.2 to −1.7, item 2), with the wrong sign structure for BET/SHR (observed +1.8 / +3.3). This is consistent with the review's claim that simple thermal-time models over-respond, and shows the TFT's near-zero response is not the only mismatch in the comparison set.

## 6. What this means for the proposal

1. **The observed SOS ST is supported for deciduous broadleaf forest, and the proxy is not the weak link.** PhenoCam supports the sign, magnitude (−2.3 ground vs −2.1 LCSIF at matched BDT sites) and interannual timing (r = 0.74). The sampling/smoothing chain contributes essentially no attenuation (simulation, ratio ≈ 1) but adds noise (a single-pixel 40-y ST has s.d. 0.5–1.5 d/°C), which is why the claim should be made at the PFT/zone level with bootstrap intervals (item 2).
2. **Nothing is validated for EOS.** Ground autumn dates are not reproduced (r_within ≈ 0.1) with either PhenoCam or NPN. Any chapter claim that rests on observed EOS sensitivity (including the water/EOS claims) should be labelled unvalidated until PEP725 or other autumn data are added.
3. **No validation for evergreen needleleaf, deciduous needleleaf, grass, shrub or broadleaf evergreen** at the sites available (n ≤ 7 sites with ≥5 years). The BET/SHR positive-ST finding (warmer = later green-up) is therefore LCSIF-only. PhenoCam NET (r_within 0.69 but a 17 d offset and a ratio of 0.29 [−0.18, 0.91]) is the one hint that evergreen GCC and SIF differ in what they track.
4. **Benchmark claims.** A head-to-head table can now be written for NET, NDT, BDT and BET with the same events, but (a) "4GST" is an OLS extrapolator and "PhenoFormer" is an MSE sequence model without future weather, so the text should call them what they are; (b) ThermalTime as stored is invalid, only the thermal-time positive control is usable; (c) PhenoFormerSeq matches the TFT on window timing, so the TFT is not demonstrably better on this metric; (d) every learned model, TFT included, has an SOS response to warming 5–15× smaller than observed.
5. The comparison does not change the item-1 conclusion that daily-lead skill is partly spline smoothness. The benchmarks share the same product, so the timing errors above are relative, not absolute.

## 7. Caveats
- Ground ST uses ERA5 pixel T for both ground and LCSIF, so any T error cancels in the ratio but means ST_ground is itself an apparent sensitivity (error in T dilutes it by an unknown amount). Real site T would raise both.
- PhenoCam GCC at an ROI is a canopy-level, upper-canopy greenness; its SOS is earlier/later than SIF half-amplitude by an unknown biome-dependent offset. The BDT bias is small (+1 d); the NET offset is +17 d.
- NPN data are volunteer observations of individual plants; the first-yes is conditional on sampling frequency and the dominant bias is positive for earlier events.
- 14-year overlap only; per-site OLS slopes have large s.e., so the pooled within-site slope with site-bootstrap is used. Bootstraps resample sites, not years.
- The matching of a ground year to an LCSIF season-year is by calendar year; this is fine for NH spring SOS but not tested for the Southern Hemisphere (excluded with lat ≥ 25).
- Benchmark-window alignment is by row order (no ids). I could verify alignment through skill and N only.
- The PhenoFormer perturbation is a linear +1 °C in scaled-through-the-scaler space on the history only (the model takes no future input).

## 8. How to rerun
```
sbatch bashscripts/review/item6/download.sh --only phenocam ; sbatch ... --only npn   # public data
python -I scripts/review/item6/ground_build.py                                          # ground tables + pixel matches (srun)
sbatch bashscripts/review/item6/lcsif_pixels.sh PFT ground --locs-file data/review/item6/locs/PFT_ground.txt   # per PFT
sbatch bashscripts/review/item6/compare_ground.sh        # LCSIF vs ground -> ground_vs_lcsif.csv
sbatch bashscripts/review/item6/sim_attenuation.sh       # simulation -> sim_attenuation.csv
sbatch bashscripts/review/item6/score_windows.sh PFT     # BDT needs --mem=170G
sbatch bashscripts/review/item6/pf_perturb.sh PFT        # CPU only
```
Nothing is still running. Stale/failed jobs (first score_windows runs: scaler key; first lcsif_pixels BET/GRA: JSON key; first pf_perturb: doys shape) were fixed and rerun; the files in the output directories come from the successful reruns (checked by timestamp).

## 9. Figures (`scripts/review/item6/make_figures.py` → `data/review/item6/figures/`)
1. `fig1_ground_vs_lcsif_dates`: LCSIF vs ground dates and within-site anomalies (PhenoCam BDT SOS/EOS, NPN BDT SOS). 2. `fig2_sensitivity_ground_vs_lcsif`: ST from ground vs LCSIF and the ratio with site-bootstrap CIs. 3. `fig3_sim_attenuation`: the 36 attenuation scenarios. 4. `fig4_benchmark_timing_skill`: median timing error and pooled r of TFT / PhenoFormerSeq / LSTM / 4GST / persistence / ThermalTime on identical events. 5. `fig5_model_sensitivity_vs_observed`: PhenoFormerSeq +1 °C response and thermal-time ST vs observed. 6. `fig6_bdt_model_vs_observed_timing`: BDT predicted vs observed (LCSIF) crossing day per model, all test events. 7. `fig7_phenocam_model_anomalies` (+ `fig7b_…_only`, and `_ev2` variants): PhenoCam vs model dates / within-site anomalies at PhenoCam BDT pixels (section 10).

## 10. PhenoCam vs TFT / PhenoFormerSeq / 4GST at the PhenoCam pixels (`phenocam_cache.py`, `phenocam_models.py`; `bashscripts/review/item6/phenocam_{cache,models}.sh`)
Stored predictions cover only 2002-2010 (11 PhenoCam matches), so the models were re-run at the 118 PhenoCam BDT pixels (171 sites, ≥25°N, own-PFT) for 2008-2021. Inputs are rebuilt from the original (not smfix) CSIFMETEO parquet with the scalers and feature order saved in the TFT_30 BDT pickle (soil = raw class value truncated to int32, as stored). **Validation:** the built inputs equal the pickle's test-window inputs exactly (80 windows) and the re-run TFT (Feb-2025 BDT base) and PhenoFormerSeq reproduce the stored predictions to 3.7e-5 and 2.9e-6 (`data/review/item6/phenocam_models_validation.txt`). LSTM is excluded (categorical embeddings not saved). 4GST = OLS extrapolation recomputed.
Windows (30-d nowcast, not a long-lead forecast): `--anchor smooth` (fig 7) centres the 30-d window on the observed LCSIF half-amplitude date (start = date − 15 d); `--anchor ev2` (fig 7 `_ev2`) uses the FINDINGS §2 event window. The ev2 window sits in a fixed DOY band, so window position carries almost no interannual information. Model date = window start + half-amplitude crossing day of the median forecast; `LCSIF window` is the same crossing of the observed window.
**In-sample caveat:** 2011-2021 are training years of both checkpoints (and BDT has no spatial hold-out), so only 2008-2010 are out-of-sample, which is too few (≤ 52 site-years, 0 sites with ≥ 5 years) to score on their own. The results show whether the models can represent interannual timing, not a validation.
Within-site r (sites with ≥ 5 matched years), `phenocam_models_summary_{smooth,ev2}.csv`:

| anchor | phase | LCSIF window | LCSIF smooth (ref.) | TFT | PhenoFormerSeq | 4GST |
|---|---|---|---|---|---|---|
| smooth | SOS | 0.55 | 0.68 | 0.54 | 0.56 | no crossing in 87 % |
| smooth | EOS | 0.04 | 0.10 | 0.02 | −0.01 | 0.03 (31 % no crossing) |
| ev2 | SOS | 0.37 | 0.68 | −0.09 | −0.05 | no crossing in 85 % |
| ev2 | EOS | −0.10 | 0.10 | −0.17 | −0.16 | −0.11 (47 % no crossing) |

Reading: with the window centred on the observed date the models simply inherit the anchor's skill (SOS r ≈ 0.55 for the observed window, TFT and PhenoFormerSeq alike; MAE 9 d, bias +2 to +3 d); they add no interannual information beyond the anchor. With a fixed window the observed crossing day still tracks PhenoCam (r = 0.37), but neither model does (r ≈ −0.1). EOS is not reproduced by anything (r ≈ 0). 4GST rarely crosses. TFT and PhenoFormerSeq are indistinguishable here.
