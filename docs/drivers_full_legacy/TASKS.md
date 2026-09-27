# TASKS — drivers_full_legacy

## Status
_(overwritten by the loop at the end of every iteration; history is in `PROGRESS.md`, choices in `DECISIONS.md`)_
- Phase: not started. Plan and decisions D1–D5 finalised 2026-09-27.
- Running jobs: none.
- Next: task 0.1.
- Open BLOCKED items: none.

Read only the phase you are working on. PLAN § references point into `PLAN.md`; read just that section.
Legend: `[ ]` todo · `[~]` job submitted (job ID) · `[x]` done and verified (≤ 2 lines of evidence) · `[!]` blocked.
Paths: code `scripts/drivers_full_legacy/` (helpers in `_common.py`), sbatch `bashscripts/drivers_full_legacy/`,
tests `tests/test_drivers_full_legacy.py`, heavy `$WORK=/burg-archive/glab/users/al4385/analysis/drivers_full_legacy/`,
small `data/drivers_full_legacy/`, logs `outputs/drivers_full_legacy/logs/`. All other rules are in PROMPT.md.

---

## Phase 0 — provenance and setup (no heavy compute)
- [ ] 0.1 Create `data/drivers_full_legacy/`, `outputs/drivers_full_legacy/logs/`, and `$WORK` with its
      subdirectories (`events/ climatology/ model_out/ samples/ occlusion/`). Confirm `outputs/` is already
      gitignored. Add nothing large to the repo.
- [ ] 0.2 Start `NOTES.md` (in this directory) with a provenance table, one row per PFT: data pickle,
      event index files and counts, the pickle's `feature_map` (historical + future order, read in a
      SLURM job and never guessed), checkpoint, checkpoint's expected input order (from
      ABLATION_FINDINGS/EXPLORATION_NOTES, and re-verified in B3), and the existing prediction file
      for the NEW-order regression checks (SHR: whichever file `bashscripts/analysis/sensitivity_SHR.sh`
      used, i.e. `SHR_20251023.pkl`, and check it; BDT: `predictions_40years/BDT/...TFT30_40years.pkl`).
      Also write this table to `data/drivers_full_legacy/validation/provenance.csv`.
- [ ] 0.3 Check that event indices align with their pickle: for 2,000 random SOS and 2,000 random EOS
      indices per PFT, the observed target (`data_sets.test.target`, inverse-scaled or not, since
      monotonicity is scale-free) must be strictly increasing (SOS) or strictly decreasing (EOS) over
      the 30 days in ≥ 99% of cases. Record the rates. If a PFT fails, mark it `[!]` and add a BLOCKED
      entry, because its indices point into a different pickle.
- [x] 0.4 Decisions D1–D5 (PLAN §10) resolved by the author on 2026-09-27: D3 keeps the §6.4 rule as
      written, and D1, D2, D4, D5 use the defaults. Do not revisit.

## Phase A — event extraction (SLURM, mem768)
- [ ] A1 `extract_events.py --pft P`. Load the pickle and slice the test split at `unique(SOS ∪ EOS)`
      indices: `historical_ts_numeric`, `future_ts_numeric`, static numeric/categorical, any
      categorical time series, `target`, `id`. Store the column **name lists** and `phase` flags.
      Write to `$WORK/events/{P}_events.npz`. Also store the train-split scaled means (reuse
      `scaled_means`).
- [ ] A2 In the same job, build the DOY climatology (PLAN §6.2). Per location, reconstruct each day's
      date from `id` (forecast start) minus lag, take one sample per location-year to cover all DOYs
      without overweighting overlapping samples, average per (location, DOY, met variable) over
      years, then apply a 31-day circular smoothing. Write `$WORK/climatology/{P}_doy_climatology.npz`
      and log per-location year counts.
- [ ] A3 `bashscripts/drivers_full_legacy/extract_events.sh <PFT>`. Run SHR first and verify: event
      count = |SOS ∪ EOS|, shapes, names match `feature_map`, climatology has no NaN for locations
      with ≥ 5 years.
- [ ] A4 Run the other five PFTs. Verify each the same way.

## Phase B — baseline inference on events
- [ ] B1 `infer_events.py --pft P --weights W --expected-order {OLD,NEW}`. Reorder input columns by
      name into the checkpoint's expected order (hist and future separately), strict-load, and run a
      forward pass in batches of 128. Save to `$WORK/model_out/{P}_{phase}_outputs.npz`: per event the
      historical attention profile α (365), concurrent profile γ (30), `historical_selection_weights`
      (365×7) + name list, `future_selection_weights` (30×6) + name list, `predicted_quantiles`,
      `target`, `id`. Use float32. Add float16 only if size forces it, and record that.
- [ ] B2 Unit tests (`tests/test_drivers_full_legacy.py`) with tiny synthetic arrays:
      name-based reordering for OLD↔NEW including `sm` alias; history/concurrent key split; per-day
      renormalisation excluding CSIF; each estimator variant against hand-computed values (e.g. a
      one-hot attention reproduces that day's shares; uniform attention = `uniform_null`); lag-band
      bookkeeping; aggregation event → location-year → location; bootstrap determinism with a
      fixed seed. `pytest tests/test_drivers_full_legacy.py -v` must pass.
- [ ] B3 SHR smoke run (≤ 2,000 events). Checks: (a) with NEW order, outputs match the existing SHR
      prediction pickle for the same indices (PLAN §5.1 criteria) and the numbers are recorded;
      (b) the median-forecast R² vs target on events is recorded.
- [ ] B4 Full SHR inference. Then the full runs for GRA and BDT (NEW order; run the same regression
      check vs existing preds on 2,000 events).
- [ ] B5 BET, NET, NDT (OLD-order checkpoints, reordered inputs). Checks: event R² ≥ 0.7 (else
      BLOCKED), and on 2,000 events the *mis-ordered* configuration reproduces the existing broken
      prediction file. This documents P2. Record both R² values.

## Phase C — M1 estimators and aggregation
- [ ] C1 `compute_drivers.py --pft P --phase S`. From `model_out`, compute per event every variant in
      PLAN §5.3 (primary `aw_above_uniform`, `aw_all`, `aw_hdr50`, `aw_hdr80`, `aw_excess`,
      `aw_legacy_only`, `uniform_null`, `perm_null` with fixed seed and PFT × phase × 10°-lat strata,
      `legacy_maxwindow_byname`, `legacy_maxwindow_index`, `aw_post_norm`), plus the lag-band masses and
      per-band composition, driver mean lags, CSIF share, historical attention mass, concurrent
      composition, and the skill-gate flag (sign of forecast slope = sign expected for the phase).
      Write `$WORK/samples/{P}_{phase}_samples.parquet`.
- [ ] C2 Aggregation (PLAN §5.4): event → location-year → location, ≥ 5-year support, B = 1000
      year-bootstrap (seeded) giving CIs and dominance confidence. Write the primary CSV
      `data/drivers_full_legacy/{P}_drivers_{S}.csv` (schema in PLAN §8), plus `variants/`, `lag/` and
      `concurrent/`. Append coverage (locations before and after the support rule) to
      `validation/coverage.csv`. If > 20% of locations are dropped, BLOCKED.
- [ ] C3 **Regression test for P1** (PLAN §7.6). For SHR and BDT, `legacy_maxwindow_index` aggregated
      the old way (SHR: mean over samples; BDT: median over samples) must reproduce
      `data/drivers_data/SHR_drivers_{SOS,EOS}.csv` and `BDT_drivers_2_{SOS,EOS}.csv` for the same
      locations (median |Δ| ≤ 1e-3). If it doesn't, stop and find out why before going further. Then
      report how far `legacy_maxwindow_byname` and the primary estimator move the shares.
- [ ] C4 Run C1–C2 for SHR, check the outputs make sense (shares sum to 1, CIs contain the point
      estimates, sensible coverage), then run the other five PFTs × 2 phases
      (`bashscripts/drivers_full_legacy/compute_drivers.sh`, array over PFT × phase).
- [ ] C5 Null diagnostics (PLAN §5.5): per PFT × phase, the distribution of |primary − uniform_null|
      and of (primary − perm_null). Write `validation/null_diagnostics.csv`. State plainly in NOTES
      whether attention weighting changes the composition materially.

## Phase D — M3 occlusion validation
- [ ] D1 `occlusion_events.py --pft P --phase S --chunk k --n-chunks K --reference {clim,mean}`.
      Implement the 21 perturbed passes + baseline in PLAN §6.2 (by name, hist + concurrent),
      statistics f and β, and output `Δ` per (group, band) to `$WORK/occlusion/...parquet`. Unit-test
      band masking and the by-name replacement on synthetic arrays (extend `tests/`).
- [ ] D2 SHR: first a 2,000-event smoke test. Check that the baseline pass equals B-phase predictions
      exactly and that the time per event matches the plan. Then run full SHR with the climatology
      reference, plus a 10% stratified subsample with the mean reference.
- [ ] D3 The other PFTs (array jobs sized to about 4 h per task). Check `sacct` exit codes. Resubmit
      only failed chunks.
- [ ] D4 Aggregate occlusion importances with the same aggregator as C2 →
      `data/drivers_full_legacy/occlusion/{P}_drivers_{S}.csv` (same schema, so the same map code
      works). Report non-additivity and sign summaries.
- [ ] D5 Agreement analysis (PLAN §6.4): per PFT × phase, Spearman ρ per composite, dominant-composite
      agreement, Cohen's κ, and lag-band ranking agreement → `validation/agreement_M1_M3.csv`. Apply
      the pre-registered decision rule and record, per PFT × phase, whether M1 is corroborated and
      therefore which maps are the headline.

## Phase E — maps
- [ ] E1 `generate_maps.py` mirroring `scripts/generate_all_driver_maps.py`: same generator settings,
      global stretch bounds refitted on the new primary CSVs and logged, the same 28 output files in
      `data/drivers_full_legacy/maps/`, and the legend in `maps/legend/`. Also write a low-res contact
      sheet (`maps/contact_sheet.png`, new next to old for each map), view only that image, and
      confirm nothing renders blank and that the coverage looks like the old maps.
- [ ] E2 Occlusion maps (same format) → `maps/occlusion/`.
- [ ] E3 Supplementary maps → `maps/supplementary/`: raw (unstretched) primary fractions; dominance
      confidence; lag-band attention mass (4 panels) and driver mean lags; concurrent composition and
      mass; M1-vs-M3 disagreement; old-vs-new dominant-composite change (old = `data/drivers_data`,
      labelled as affected by P1/P2); skill-gate-masked version. Use perceptually uniform colormaps
      for scalar maps, and apply BH-FDR (q = 0.10) wherever pixels are marked significant.
- [ ] E4 `bashscripts/drivers_full_legacy/generate_maps.sh`. Run it and check that all expected files
      exist.

## Phase F — robustness and write-up
- [ ] F1 Robustness tables (PLAN §7.1–7.5): variant agreement, in- vs out-of-sample years, epochs,
      skill gate, aggregator/support. Write them under `validation/`.
- [ ] F2 `FINDINGS.md`. It must contain: goal; data and event definition; the P1–P7 problems and how
      they were handled; the M1 method with equations and justification; the M3 method; the
      pre-registered choices and any deviations; results per PFT × phase (composite shares with
      CIs, dominant-driver pixel fractions from raw fractions, lag profiles, concurrent mass);
      validation (agreement table and decision-rule outcome); robustness; limitations (attention on
      LSTM states, in-sample years, selection-weight invariance, NET skill caveat); a draft Methods
      paragraph; a summary of every `DECISIONS.md` entry that affects results; and the list of figures with paths. Every number must trace to a file in
      `data/drivers_full_legacy/validation/` or `NOTES.md`.
- [ ] F3 Final check: every task above is `[x]` with evidence (or `[!]` with a justified BLOCKED
      entry); `pytest tests/test_drivers_full_legacy.py` passes; `ruff check scripts/drivers_full_legacy`
      is clean; nothing was committed; `data/drivers_data/` is unchanged (`git status` shows no
      modifications there beyond what existed at the start — see NOTES for the starting snapshot).

## Phase G — optional stretch (only once F is done)
- [ ] G1 Integrated Gradients on a stratified ~20k-event subsample (PLAN §6.3), then compare the
      day-resolved IG to `P_i(τ, v)`.

---

## Deviations from pre-registered choices
(none yet — record date, what changed, why, and confirm it was written before the affected result
was computed)

## BLOCKED
(none yet)
