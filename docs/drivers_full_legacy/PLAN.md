# drivers_full_legacy — attention-weighted, time-resolved driver attribution for SOS and EOS

Status: **plan, not yet executed** (written 2026-09-27). **This is a reference document. Loop agents: read
only the section a task cites** (`grep -n '^## \|^### ' PLAN.md`, then `sed -n` that range). Companion files: `TASKS.md` (checklist the
Ralph loop works through), `PROMPT.md` (the loop prompt). Findings will go in `FINDINGS.md`, working
notes in `NOTES.md`.

---

## 1. Goal

The current driver maps (`data/drivers_data/maps/`) say which climate driver the TFT relies on for
Start of Season (SOS) and End of Season (EOS) by looking at **one** 30-day window: the contiguous
window of maximum summed attention in the 365-day history. Inside that window they take the median
variable-selection weight of each driver.

This analysis replaces that single-window reading with one that uses the **whole attention
distribution over the 365-day past** ("full legacy"):

> For every SOS/EOS forecast, take each historical day the decoder attends to above chance. At each
> of those days, find the relative importance of every meteorological driver. Combine them, weighted
> by how much attention each day gets. The result is a per-event driver composition, and a
> lag profile saying *when* in the past each driver mattered.

Deliverables:

1. Per-location driver CSVs in `data/drivers_full_legacy/`, with the same schema as
   `data/drivers_data/*_drivers_{SOS,EOS}.csv` so the existing map code can read them.
2. The same maps as `data/drivers_data/maps/` (6 PFTs × SOS/EOS + `All_PFTs_{SOS,EOS}`, PNG + SVG,
   plus the ternary legend), written to `data/drivers_full_legacy/maps/`.
3. Supplementary products: lag profiles per driver, the split between antecedent and concurrent
   attention, per-pixel uncertainty/confidence, and a **model-agnostic validation** of the new
   estimator (§6).
4. `FINDINGS.md`: methods text, validation results, robustness checks and a clear statement of what
   the maps can and cannot support. It has to be fit for a peer-reviewed journal.

All SLURM jobs follow the existing `bashscripts/` convention (an sbatch wrapper calling a Python
entry point in `scripts/`).

---

## 2. How the current driver data and maps are produced (verified by reading the code)

### 2.1 SOS/EOS event definition (the "samples" the drivers are computed for)
`debugging/save_sos_eos_indices_{PFT}.py` (run by `bashscripts/drivers/generate_sos_eos_indices_*.sh`):
- It reads a per-PFT analysis dataframe (observed CSIF per location/day) and walks it in
  non-overlapping 30-day batches.
- A batch is an **SOS event** if its start DOY is inside the latitude-specific SOS DOY window
  (`debugging/debugging_subset_data/sos_eos_{PFT}_merged_smoothed.json`) **and** observed CSIF
  increases strictly on every day. An **EOS event** is the same with the EOS window and a strict
  decrease.
- Each batch is mapped to a *prediction index* by matching `(location, start date)` to
  `data_sets.test.id[:, 0]` of the TFT pickle. So an event = one 30-day forecast window whose target
  is a monotonic green-up or brown-down.
- Index files: `data/SOS_EOS_indices/{BET,GRA,NDT,NET,SHR}/{SOS,EOS}_pred_indices_{PFT}.npy`. BDT uses
  `debugging/debugging_subset_data/{SOS,EOS}_pred_indices_BDT_1.npy`, which indexes the merged
  `TFT_30_40years/sorted_merged_BDT_1982_2021.pkl`.
- Event counts (SOS / EOS): BDT 163,317 / 164,603 · BET 182,293 / 154,007 · NET 140,651 / 154,781 ·
  NDT 44,180 / 63,347 · GRA 91,353 / 119,063 · SHR 75,314 / 67,773. **About 1.42 M events in total.**

### 2.2 Per-sample driver weights (the "max-attention-window" method)
`scripts/extract_epoch_drivers.py` (`bashscripts/analysis/sensitivity_{PFT}.sh`) and
`debugging/driver_data_modified.py` (BDT, via `bashscripts/drivers/extract_drivers_refactored.sh`)
do the same thing:
1. `att = mean(attention_scores[i], axis=0)`: average the (30 forecast queries × 395 keys) attention
   matrix over queries (`extract_epoch_drivers.py:177-193`).
2. Slide a 30-day window over the history and keep the start with the largest summed attention
   (argmax, capped at start 335).
3. Inside that window, take the **median over the 30 days** of
   `historical_selection_weights[i, ws:ws+30, 1:7]` and label columns 1..6 as
   `tmin, tmax, rad, precip, photo, sm` (`extract_epoch_drivers.py:254-258`;
   `get_driver_map_data_refactored.py:269-275`).

### 2.3 Per-location aggregation → `data/drivers_data/*.csv`
- BDT (`BDT_drivers_2_{SOS,EOS}.csv`): **median** over samples per location
  (`driver_data_modified.py`), with an extra `max_att_start` column.
- BET/NET/NDT/GRA/SHR (`{PFT}_drivers_{SOS,EOS}.csv`): `scripts/build_driver_csvs_from_sensitivity.py`
  takes the **mean** of the per-sample rows in `data/sensitivity_data/{PFT}_yearly_drivers.csv`. Its
  docstring says it averages across years, but it averages rows (samples) directly.
- **Event definitions differ across PFTs.** `sensitivity_{GRA,BET}.sh` use `--detect-csif` (a
  slope threshold on the flattened target) instead of the `.npy` index files. BDT uses the older
  `TFT_30_40years` data and predictions; the other PFTs use `TFT_40_overlapping_samples` and
  `pred_40year_moresamples`.

### 2.4 Maps → `data/drivers_data/maps/`
`scripts/generate_all_driver_maps.py` → `DriverMapGenerator` (`src/phenofusion/analysis/generate_driver_maps.py`):
1. `compute_raw_fractions` (l.233): per (lat, lon) mean over rows, then composites
   temp = tmin+tmax, sol = rad+photo, p = precip+sm (l.260-262), renormalised to sum to 1.
2. `compute_global_stretch_bounds` (l.267): 2nd–98th percentile bounds pooled over all 12 PFT×phase
   files. `_contrast_stretch` (l.176) rescales each composite to those bounds and renormalises. This
   is a **visualisation-only** transform, and any statistic must be computed on raw fractions.
3. Gridded to 0.25° (720×1440). `_fill_nan_gaps` (l.116) fills empty cells with the median of up to
   10 neighbours within 0.25°.
4. `colorize_data` (l.371): barycentric blend of solar=#D4AF00, precip=#00E5D4, temp=#F02FE0. Uses
   PlateCarree, crops below −60°, dpi 300, PNG + SVG. PFT latitude filters are in `lat_filters`
   (l.71). The key `BDT` has no filter.
5. Outputs: `{BDT,BET,NET,NDT,GRA,SHR}_{SOS,EOS}.{png,svg}` and `All_PFTs_{SOS,EOS}.{png,svg}`
   (28 files), plus `data/drivers_data/legend/ternary_legend.{png,svg}`.

---

## 3. Known problems this analysis must not inherit

These were found and verified by the attention-validation loop
(`docs/ralph_attention_validation/ABLATION_FINDINGS.md`, 2026-09-13). They are **not fixed in the
repo**, and each one would invalidate the new maps if carried over.

| # | Problem | Consequence | How this plan handles it |
|---|---|---|---|
| P1 | Selection-weight columns are read by **hard-coded index 1..6**. For NEW-order pickles (`[tmin, tmax, precipitation, radiation, photoperiod, swvl1, sif_clear_inst]`, which covers all TFT_40 pickles and merged BDT) this drops tmin, labels precipitation as "tmax", photoperiod as "precip", swvl1 as "photo" and **past CSIF as "sm"**. | The current "water" share for SHR/GRA/BDT is mostly the autoregressive CSIF weight. | Select every column **by name** from the pickle's `feature_map`. Alias `sm ≡ swvl1`. Unit-test the mapping. |
| P2 | The NDT, NET and BET checkpoints were trained on **OLD-order** inputs but run on NEW-order TFT_40 pickles to produce `pred_40year_moresamples/{NDT,NET,BET}_1982_2021.pkl` (forecast R² of −20, −25 and −425). | Attention and selection weights for those PFTs come from a broken model. | **Do not use the existing prediction pickles.** Re-run inference on the event samples, with inputs reordered by name into the order each checkpoint expects (§5.1). |
| P3 | Selection weights are close to **input-invariant**: across-sample std ≤ 0.015 for tmin, tmax, precip, photoperiod. Ablating temperature moves the weights by < 2e-3 while it changes forecasts materially. The ranking of drivers by selection share disagreed with ablation importance in 5 of 6 PFTs. | Any estimator built only on selection weights, including the one proposed here, risks mapping learned per-variable priors rather than per-event importance. | The plan requires a **model-agnostic validation** (§6), null models (§5.5), and a decision rule fixed in advance for which estimator backs the headline maps (§6.4). |
| P4 | Attention peaks in the last 30 days (window start 335) for almost all BDT/GRA/SHR events. | A single max window reduces to "the most recent month", so legacy effects are invisible. | The full-distribution estimator plus lag-band decomposition (§5.3) keeps both recent and legacy attention. |
| P5 | `scripts/get_attention_mean_data.py:174-180` averages attention over all 395 keys, which **includes the 30 concurrent (future) keys**. | Mean "historical" day is contaminated by concurrent attention. | Split keys explicitly: history = 0..364, concurrent = 365..394. |
| P6 | Aggregation is inconsistent across PFTs (median vs mean; samples vs years; different event definitions). | Differences between PFTs partly reflect processing. | One event definition (the `.npy` indices) and one hierarchical aggregator (§5.4) for every PFT. |
| P7 | Inputs are **QuantileTransformer**-scaled to [0, 1], not z-scored. | A value of 0 is the record minimum, not neutral. | Occlusion references are climatology or train mean in scaled space, never 0 (§6.2). |

The analysis also has to be explicit about what the TFT quantities *are* (checked in
`tft_torch/tft.py`):
- `attention_scores[i]` has shape (30 queries × 395 keys) and is averaged over the 4 heads. Each row
  is a softmax over keys, with a causal mask so query q only sees history plus future steps ≤ q.
- `historical_selection_weights[i]` has shape (365 × 7). Each row is a softmax over the 7
  historical inputs at that day: how the input embedding **for that day** was composed.
- `future_selection_weights[i]` has shape (30 × 6): the same for the known concurrent meteorology.
- Attention acts on the **LSTM-encoded, statically enriched** sequence. The representation at day τ
  therefore carries information from days ≤ τ, so attention at τ is not strictly "attention to
  day τ's inputs". The "attention × selection" composition in §5 is a model-internal information
  routing summary, not a causal decomposition. The paper must say so, and it is one reason §6
  exists.

---

## 4. Estimand: what question the maps answer

We distinguish two estimands and keep them apart in every figure and sentence:

- **E1: model reliance (routing).** Of the meteorological information the model draws on to
  forecast CSIF during an SOS/EOS window, what share comes from temperature, solar and water inputs,
  and from how far back? Estimated by attention × variable selection (§5, "M1").
- **E2: sensitivity to interannual anomalies (counterfactual).** If a driver's weather in a given
  lag band had been climatological, how much would the forecast of the green-up/brown-down change?
  Estimated by occlusion against climatology (§6, "M3"). This is closest to the ecological question
  "which climate factor controls the *timing* of SOS/EOS".

Photoperiod has no interannual variability, so it has zero weight under E2 by construction. Under
E1 it can still carry routing weight because it encodes the seasonal clock. This is expected, and
the maps' "solar" composite has to be described accordingly.

Every attribution here describes **what the fitted TFT uses**. Ecological claims need (i) agreement
between E1 and E2, and (ii) consistency with independent phenology literature. The TFT_40 "40-year
test mode" sets include the model's training years (train <1992 and ≥2011), so attribution is
partly in-sample. That is acceptable for describing model reliance, but §7 checks that
out-of-sample years (1992–2010) give the same picture.

---

## 5. Method M1: attention-weighted, time-resolved driver composition

### 5.1 Inference on event samples (replaces reading the big prediction pickles)
For each PFT, extract the event samples (SOS + EOS indices) from the TFT pickle once. Then run the
checkpoint on just those samples with **inputs reordered by name to the checkpoint's training
order**, and `load_state_dict(strict=True)`. Save per event: `attention_scores`,
`historical_selection_weights`, `future_selection_weights`, `predicted_quantiles`, `target`, `id`.

This fixes P2, uses one uniform code path for all PFTs, and avoids loading 70–277 GB prediction
pickles. Cost is about 4 ms per sample on a 32-core compute node (measured in the ablation work),
so 1.42 M events is well under 2 node-hours for a baseline pass.

Default checkpoints are the ones whose test skill the paper reports and that were verified to
reproduce the reference predictions (ABLATION_FINDINGS §4.5):

| PFT | data pickle (events indexed into its test split) | checkpoint | checkpoint input order |
|---|---|---|---|
| BDT | `data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl` | `weights_merged_BDT_1982_2021_feb2025_checkpoint.pth` | NEW (`sm`) |
| BET | `data/TFT_40_overlapping_samples/BET_1982_2021.pkl` | `weights_BET_3.pth` | OLD → reorder |
| NET | `data/TFT_40_overlapping_samples/NET_1982_2021.pkl` | `weights_NET_2.pth` (paper NET R² caveat) | OLD → reorder |
| NDT | `data/TFT_40_overlapping_samples/NDT_1982_2021.pkl` | `weights_NDT_2.pth` | OLD → reorder |
| GRA | `data/TFT_40_overlapping_samples/GRA_1982_2021.pkl` | `weights_sorted_GRA_3.pth` | NEW |
| SHR | `data/TFT_40_overlapping_samples/SHR_1982_2021.pkl` | `weights_sorted_SHR_3.pth` | NEW |

(Paths under `/burg-archive/glab/users/al4385/`. The checkpoint for each PFT is decision **D1**, §10.
The loop must record the verified input order of every checkpoint before using it.)

Acceptance checks:
- **NEW-order PFTs:** re-inferred outputs must match the existing prediction pickle on ≥ 2,000 event
  samples (max |Δ| ≤ 1e-3; the ablation work saw CPU-vs-GPU drift up to 2.5e-2 on a handful of BDT
  samples, so the check uses max |Δ| on the 99.9th percentile plus a correlation above 0.99999).
- **OLD-order PFTs:** event-sample forecast skill must be sane (R² of the median quantile vs target
  in the range of the TFT_30 test metrics, ≥ 0.7). The mis-ordered configuration has to be confirmed
  as reproducing the broken existing file.

### 5.2 Per-event quantities
Notation for event *i*: history days τ = 0..364 (τ = 364 is the day before the forecast window),
lag ℓ = 365 − τ days. Concurrent days are κ = 0..29. `A_i` is the attention matrix.

- Historical attention profile: `α_i(τ) = (1/30) Σ_q A_i[q, τ]`.
  Concurrent profile: `γ_i(κ) = (1/30) Σ_q A_i[q, 365+κ]`.
  Historical attention mass: `m_i = Σ_τ α_i(τ)`, which is in (0, 1]. `1 − m_i` is the concurrent mass.
- Normalised historical attention: `ᾱ_i(τ) = α_i(τ) / m_i`, which sums to 1.
- Meteorological driver share at day τ (**selected by name**, CSIF excluded and reported
  separately): `s_i(τ, v) = W_i[τ, v] / Σ_{v' ∈ MET} W_i[τ, v']` for v ∈ {tmin, tmax, precipitation,
  radiation, photoperiod, swvl1}. The past-CSIF share is `c_i(τ) = W_i[τ, sif_clear_inst]`.
  Renormalising **per day** gives "relative importance of each driver at that timestep", which is
  the quantity the research question asks for. The existing maps renormalise only after aggregating.
  That variant is kept as a sensitivity check.

### 5.3 The estimator
**High-attention set (primary, pre-registered):** `H_i = { τ : ᾱ_i(τ) > 1/365 }`, the days the
decoder attends to more than it would under uniform attention. This threshold needs no tuning: it
is the natural chance level for a softmax over 365 positions.

**Primary per-event composition:**

    D_i(v) = Σ_{τ∈H_i} ᾱ_i(τ) · s_i(τ, v)  /  Σ_{τ∈H_i} ᾱ_i(τ)

Composites: temp = tmin + tmax, sol = radiation + photoperiod, water = precipitation + swvl1. These
are the paper's existing groupings. Each composite is also kept at single-variable resolution.

Equivalently, `P_i(τ, v) ∝ ᾱ_i(τ) s_i(τ, v)` over τ ∈ H_i is a joint **time × driver attribution
surface** for the event. `D_i` is its driver marginal. Its lag marginals give the legacy products:

- **Lag-band decomposition.** Bands are pre-registered: B1 = 1–30 d (proximate; same scale as the
  old window and the forecast horizon), B2 = 31–90 d (seasonal preconditioning, e.g. forcing and
  chilling accumulation), B3 = 91–180 d (previous season), B4 = 181–365 d (previous growing season /
  ecological memory). Per event, record the attention mass in each band, `M_i(b)`, and the
  composition within each band, `D_i(v | b)`.
- **Driver-specific attention-weighted mean lag:** `L_i(g) = Σ_τ P_i(τ, g) ℓ(τ) / Σ_τ P_i(τ, g)`,
  answering "when did temperature (or water, or solar) matter for this event".
- **Concurrent composition** (`γ_i` × `future_selection_weights`, same formula) and the concurrent
  mass `1 − m_i`. These are reported alongside, not mixed into, the historical ("legacy") maps.
  Whether they go in the headline figures is decision **D4**.

**Variants, all computed from the same saved arrays** (sensitivity analysis, §7):

| id | set of days | weights | purpose |
|---|---|---|---|
| `aw_above_uniform` (**primary**) | ᾱ > 1/365 | ᾱ | as above |
| `aw_all` | all 365 | ᾱ | threshold-free expectation |
| `aw_hdr50`, `aw_hdr80` | smallest set holding 50% / 80% of ᾱ mass | ᾱ | highest-density-region thresholds |
| `aw_excess` | ᾱ > 1/365 | ᾱ − 1/365 | weights above chance only |
| `aw_legacy_only` | above-uniform, restricted to lag > 30 d | ᾱ | removes the recency peak (P4) |
| `uniform_null` | all 365 | 1/365 | what attention adds over a plain time average |
| `perm_null` | as primary | ᾱ of a random other event in the same PFT × phase × 10° lat band | tests whether attention and selection co-vary meaningfully |
| `legacy_maxwindow_byname` | contiguous 30-d max window (≤335) | median | old method with **correct labels**: isolates the effect of P1 |
| `legacy_maxwindow_index` | same | median, **index 1..6 labels** | reproduces the old CSVs, used as a regression test |
| `aw_post_norm` | primary set | ᾱ, raw W renormalised after aggregation | matches the old normalisation order |

### 5.4 Aggregation (identical for every PFT)
1. **Event → location-year**: arithmetic mean of `D_i` over the events of a location whose forecast
   window starts in that calendar year. Consecutive monotonic windows in one season are
   pseudo-replicates and must not be weighted as independent.
2. **Location-year → location**: arithmetic mean over years. Shares stay closed (they sum to 1). A
   closed geometric mean (Aitchison) is computed as a sensitivity check.
3. Minimum support (pre-registered): a location is mapped only if it has events in ≥ 5 distinct years
   for that phase. Coverage lost to this rule is reported per PFT. If > 20% of locations are dropped
   for any PFT × phase, the loop stops and asks the author (a `BLOCKED` entry).
4. **Uncertainty**: a year-level bootstrap per location (B = 1000, resampling location-years) gives
   95% percentile CIs for each composite share, plus **dominance confidence**: the fraction of
   bootstrap replicates in which the location's dominant composite matches the point estimate.
5. All counts and summary statistics (e.g. "% of pixels temperature-dominated") are computed on
   **raw** location-level fractions, never on contrast-stretched or gap-filled grids.

### 5.5 Null models and what they tell us
- If `aw_above_uniform ≈ uniform_null` (median |Δshare| < 0.01 across locations), attention weighting
  adds little. The composition is then dominated by the selection-weight priors (P3), and M1 cannot
  support spatial claims on its own. This outcome has to be reported, not hidden.
- `perm_null` asks whether the days a given event attends to have systematically different driver
  shares than random days. Report `D_i − D_i^{perm}` maps and their distribution.

---

## 6. Validation: model-agnostic, time-resolved occlusion (M3)

This is required because of P3. The prior ablation showed selection shares do not rank drivers the
way ablation does, so the M1 maps cannot go to a journal without an independent check.

### 6.1 Forecast statistic being attributed
For event i, compute `f_i` = mean of the median-quantile forecast over the 30-day window (the level
of the green-up/brown-down). The secondary statistic is `β_i` = OLS slope of the median forecast
over the window (the rate of green-up/brown-down, i.e. the phenological transition itself).

### 6.2 Occlusion design
For each driver group g ∈ {temp, sol, water} and each band b ∈ {concurrent (future window), B1, B2,
B3, B4}, replace the group's inputs **by name** inside band b with a reference series. Then compute
`Δ_i(g, b) = f_i − f_i^{(g,b)}` (and the same for β). Also run the whole-history occlusion
(B1–B4 together) and the whole-record occlusion (history + concurrent) per group.
That is 3 × 5 + 3 × 2 = 21 perturbed forward passes plus the baseline, about 90 ms per event on a
node, so **~35 node-hours for all 1.42 M events**. This fits comfortably in SLURM arrays on `short`.

References:
- **Primary: location-specific day-of-year climatology** in scaled space. For each location, take
  the mean over years of each variable at each DOY, smoothed with a 31-day circular moving mean, and
  compute it from the same pickle's historical arrays. The perturbation is "this weather was
  normal". It stays close to the data manifold, avoiding the extrapolation problem of constant or
  permuted inputs (Hooker et al. 2021). It targets estimand E2.
- **Secondary: train-split scaled mean** (as in the ablation study, for continuity), run on a
  stratified 10% subsample.

Group importance is `I_i(g) = |Δ_i(g, whole-record)|`, and the lag-resolved importance is
`|Δ_i(g, b)|`. Shares are `I_i(g) / Σ_g' I_i(g')`. Report the sign of Δ separately. Non-additivity
(`Σ_b Δ(g, b) ≠ Δ(g, all)`) is reported as an interaction diagnostic, not hidden. Aggregation and
bootstrap are the same as §5.4.

### 6.3 Optional (stretch): Integrated Gradients
IG (Sundararajan et al. 2017) of `f_i` w.r.t. every (day, variable) input, with the climatology
baseline and 32 steps, on a stratified subsample of about 20k events. It gives a day-resolved
attribution to compare with `P_i(τ, v)` directly. Only do this after everything else is done.

### 6.4 Decision rule (pre-registered; fixed before any map is looked at)
For each PFT × phase, compare the M1 primary (`aw_above_uniform`) and M3 (climatology occlusion)
location-level composite shares:
- Spearman ρ across locations, for each composite;
- dominant-composite agreement and Cohen's κ;
- agreement of the lag-band ranking (which band carries the most attention mass / occlusion effect).

**M1 is "corroborated" for a PFT × phase if κ ≥ 0.40 (moderate agreement, Landis & Koch 1977) and
ρ ≥ 0.5 for at least 2 of 3 composites.** Where it is corroborated, the M1 maps are the headline
driver maps. Where it is not, the headline maps for that PFT × phase are the M3 occlusion maps (same
format, same folder structure under `maps/occlusion/`). The M1 maps are then shown only as a
model-internal diagnostic, and `FINDINGS.md` says so explicitly. The rule is decision **D3**; it
cannot be changed after results are seen without a logged deviation.

---

## 7. Robustness and sensitivity (all reported in FINDINGS.md)
1. The estimator variants in §5.3: dominant-composite agreement with the primary per PFT × phase,
   and maps of where they disagree.
2. In-sample vs out-of-sample years (1992–2010 vs the rest): composite shares and dominance agreement.
3. Early vs late epoch (1982–1991 vs 2012–2021) as a descriptive check only. The sensitivity-shift
   analysis is a separate project and is not redone here.
4. Model-skill gate: the share of events whose median forecast reproduces the sign of the observed
   30-day slope (rising for SOS, falling for EOS). Report per PFT, and map the results with and
   without locations where this share is < 0.5.
5. Aggregator: arithmetic vs closed-geometric mean; minimum support ≥ 1 vs ≥ 5 years.
6. Label/order audit: re-derive the old CSVs with `legacy_maxwindow_index` (must match the old CSVs
   for SHR and BDT, the PFTs whose old predictions were not broken) and show how far
   `legacy_maxwindow_byname` moves them. This quantifies P1 for the paper's revision history.
7. Multiple testing: any per-pixel test (e.g. new vs old share, significance of a lag-band
   difference) uses Benjamini–Hochberg FDR at q = 0.10 across the field (Wilks 2016). Report field
   significance, not raw stippling.

---

## 8. Outputs and layout

**Repo home quota is full (50 G/50 G on `/burg-archive/home`)**, so heavy intermediates go to glab
scratch and only small outputs go in the repo.

```
/burg-archive/glab/users/al4385/analysis/drivers_full_legacy/     # heavy, not in repo
    events/{PFT}_events.npz              # inputs for SOS+EOS event samples (by-name column lists stored)
    climatology/{PFT}_doy_climatology.npz
    model_out/{PFT}_{phase}_outputs.npz  # attention (hist+conc profiles), selection weights, preds
    samples/{PFT}_{phase}_samples.parquet   # per-event estimator values (all variants, lag bands, CSIF share)
    occlusion/{PFT}_{phase}_chunk{k}.parquet

data/drivers_full_legacy/                                        # in repo, small
    {PFT}_drivers_{SOS,EOS}.csv        # PRIMARY, same schema as data/drivers_data (+ extra columns, below)
    variants/{variant}/{PFT}_drivers_{SOS,EOS}.csv
    occlusion/{PFT}_drivers_{SOS,EOS}.csv
    lag/{PFT}_{SOS,EOS}_lag.csv        # band masses, per-band composition, driver mean lags
    concurrent/{PFT}_drivers_{SOS,EOS}.csv
    validation/agreement_M1_M3.csv, variant_agreement.csv, skill_gate.csv, coverage.csv, provenance.csv
    maps/                              # same 28 files as data/drivers_data/maps (+ legend/)
    maps/occlusion/, maps/supplementary/   # confidence, lag, concurrent, old-vs-new, disagreement maps
```

Primary CSV columns: `location, latitude, longitude, hist_tmin, hist_tmax, hist_rad, hist_precip,
hist_photo, hist_sm` (shares that **sum to 1**, the six met drivers only). These are enough for
`DriverMapGenerator` unchanged. The extra columns are `share_temp, share_sol, share_water` with their
`_lo`/`_hi` 95% CIs, `dominant`, `dominance_conf`, `n_events`, `n_years`, `csif_share`,
`hist_attention_mass`, `mass_B1..mass_B4`, `mean_lag_temp`, `mean_lag_sol`, `mean_lag_water`.
Note that `compute_raw_fractions` averages **all** numeric columns per (lat, lon). Extra columns are
harmless there, but the map script must pass only the six driver columns if that ever changes.

Maps: a new `scripts/drivers_full_legacy/generate_maps.py` mirrors `scripts/generate_all_driver_maps.py`
(same `DriverMapGenerator`, PlateCarree, `crop_lat=-60`, dpi 300, PNG + SVG, pooled global stretch
bounds) with DATA_DIR/OUT_DIR pointing here, and writes its own ternary legend to `maps/legend/`.
Stretch bounds are refitted on the new data, and the bounds used are logged. Supplementary maps also
include **raw (unstretched) fraction** versions, so readers can see the stretch is presentation only.

---

## 9. Compute plan (SLURM)
Everything goes in `bashscripts/drivers_full_legacy/*.sh`, calling `scripts/drivers_full_legacy/*.py`.
Use `#SBATCH --account=glab --partition=short`, `module load anaconda` (torch, tft_torch and omegaconf
come from `~/.local`; CUDA is unusable, so run on CPU), and logs to
`outputs/drivers_full_legacy/logs/%x_%j.out` (create the directory before `sbatch`).

| step | resources | est. time |
|---|---|---|
| event extraction + climatology (per PFT; loads 15–65 GB pickles) | `-c 2 --mem-per-cpu=160G -C mem768` | 0.5–2 h each |
| baseline inference on events (per PFT × phase) | `-c 32 --mem=160G` | < 1 h each |
| estimators + aggregation + bootstrap (per PFT × phase) | `-c 8 --mem=64G` | < 1 h each |
| occlusion (array, ~50k events per task) | `-c 32 --mem=160G --array` | ~4 h per task, ~30 tasks |
| maps | `-c 4 --mem=64G` | < 30 min |

Validate every step on **SHR** (smallest) before scaling to the other PFTs.

---

## 10. Decisions for the author — ALL RESOLVED (author, 2026-09-27: D3 as written; D1, D2, D4, D5 = defaults)
- **D1: checkpoints. RESOLVED: default accepted.** Default is the table in §5.1 (the skill-reporting checkpoints). The
  alternative, reproducing the paper's driver-era checkpoints (`weights_NDT.pth`, `weights_BET.pth`)
  with correct order, would keep continuity with the old maps but not with the reported skill.
- **D2: primary estimator. RESOLVED: default accepted.** Default is `aw_above_uniform` with the lag bands in §5.3. It must be
  fixed before results are seen, and changing it later is a logged deviation.
- **D3: headline-map rule. RESOLVED (author, 2026-09-27): keep the §6.4 rule as written.** The
  thresholds (κ ≥ 0.40 and ρ ≥ 0.5 for ≥ 2 of 3 composites) are final. Where M1 is corroborated,
  the M1 maps are the headline; otherwise the M3 occlusion maps are, and M1 is shown as a diagnostic.
- **D4: concurrent drivers. RESOLVED: default accepted.** Default is to keep the headline maps historical-only (like the old
  maps) and put concurrent composition and mass in supplementary maps.
- **D5: BDT source. RESOLVED: default accepted.** Default is merged `TFT_30_40years` + `BDT_1` indices (continuity with
  `BDT_drivers_2`). The alternative is the four `TFT_40` latitude-band pickles, which need new
  SOS/EOS indices for three bands.

---

## 11. References (verify bibliographic details before citing in the manuscript)
- Lim, B., Arık, S. Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *Int. J. Forecasting* 37, 1748–1764.
- Jain, S., Wallace, B. C. (2019). Attention is not Explanation. *NAACL*.
- Wiegreffe, S., Pinter, Y. (2019). Attention is not not Explanation. *EMNLP*.
- Serrano, S., Smith, N. A. (2019). Is Attention Interpretable? *ACL*.
- Sundararajan, M., Taly, A., Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
- Zeiler, M. D., Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV* (occlusion).
- Fisher, A., Rudin, C., Dominici, F. (2019). All models are wrong, but many are useful. *JMLR* 20.
- Hooker, G., Mentch, L., Zhou, S. (2021). Unrestricted permutation forces extrapolation. *Statistics and Computing* 31.
- Ogle, K. et al. (2015). Quantifying ecological memory in plant and ecosystem processes. *Ecology Letters* 18.
- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*. Chapman & Hall.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B* 57.
- Wilks, D. S. (2016). "The stippling shows statistically significant grid points". *BAMS* 97.
- Landis, J. R., Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics* 33.
- Piao, S. et al. (2019). Plant phenology and global climate change: current progresses and challenges. *Global Change Biology* 25.
- Fu, Y. H. et al. (2015). Declining global warming effects on the phenology of spring leaf unfolding. *Nature* 526.
- Zani, D. et al. (2020). Increased growing-season productivity drives earlier autumn leaf senescence in temperate trees. *Science* 370.
- Keenan, T. F., Richardson, A. D. (2015). The timing of autumn senescence is affected by the timing of spring phenology. *Global Change Biology* 21.
