# Ralph loop prompt — drivers_full_legacy (attention-weighted, time-resolved SOS/EOS drivers)

Repo: `/burg-archive/home/al4385/phenofusion` (branch `paper/cleanup`). Docs: `docs/drivers_full_legacy/`.
This prompt is repeated every iteration. All state lives on disk, so resume from it.

## Goal (one paragraph)
Replace the single max-attention-window driver attribution (`data/drivers_data/`) with a
full-legacy, attention-weighted one. For every SOS/EOS event, weight each historical day the model
attends to above chance (ᾱ > 1/365) by its attention, and take the per-day meteorological driver
shares from the variable-selection weights. Validate against climatology occlusion (the
pre-registered decision rule decides which maps are the headline). Write per-location CSVs to
`data/drivers_full_legacy/` and maps in the same format as `data/drivers_data/maps/` to
`data/drivers_full_legacy/maps/`. This is journal research: follow the pre-registered choices,
verify every number, and report negative results plainly.

## Key facts (you do NOT need to read the ablation docs — everything needed is here)
- Pickles use two column orders. **NEW** hist = `[tmin,tmax,precipitation,radiation,photoperiod,swvl1,sif_clear_inst]`
  (all TFT_40 + merged BDT, where soil moisture is named `sm`). **OLD** hist = `[sif_clear_inst,tmin,tmax,radiation,precipitation,photoperiod,swvl1]`.
  Future windows use the same order without CSIF. **Always select by name** from `feature_map`. Alias `sm`→`swvl1`.
- Checkpoints (`/burg-archive/glab/users/al4385/weights/pretrained_1219/`): BDT `weights_merged_BDT_1982_2021_feb2025_checkpoint.pth` (NEW),
  GRA `weights_sorted_GRA_3.pth` (NEW), SHR `weights_sorted_SHR_3.pth` (NEW), NDT `weights_NDT_2.pth` (OLD),
  NET `weights_NET_2.pth` (OLD), BET `weights_BET_3.pth` (OLD). OLD checkpoints need their inputs reordered by name.
  The existing `pred_40year_moresamples` NDT/NET/BET files are broken (mis-ordered inputs) — never use them.
- Model outputs: `attention_scores` (N,30,395), heads already averaged, causal-masked; keys 0..364 = history,
  365..394 = concurrent. `historical_selection_weights` (N,365,7) and `future_selection_weights` (N,30,6) are softmaxes over inputs.
- Inputs are QuantileTransformer-scaled to [0,1]. 0 is the record minimum, not a neutral value.
- Reuse `scripts/attention_ablation/ablation_predict.py` (`build_model`, `canon`, `scaled_means`, by-name logic) and
  `DriverMapGenerator` in `src/phenofusion/analysis/generate_driver_maps.py`.
- Env: `module load anaconda` (torch/tft_torch in `~/.local`). CPU only (CUDA is unusable). About 4 ms/sample/node on
  compute nodes; the session node is throttled, so run anything heavy via SLURM.

## Each iteration — keep it lean
1. **Orient cheaply.** Run `sed -n '/^## Status/,/^Read /p' docs/drivers_full_legacy/TASKS.md` and `squeue -u al4385`.
   Only read more if it is not already in your context (e.g. first iteration, or after a
   compaction/restart): the relevant TASKS phase, the PLAN § the task cites (use `grep -n '^## \|^### ' PLAN.md`
   then `sed -n` that section — never the whole PLAN), `tail -40 PROGRESS.md`, and `DECISIONS.md`.
2. **Work until you hit a real wait.** Do as many consecutive tasks as you can (write code → test → submit)
   within the iteration. Batch submissions: once SHR is validated, submit all remaining PFTs in one array.
   Chain dependent steps with `--dependency=afterok:<id>` so they run without you.
3. **Never end an iteration just to poll a job.** Ending re-triggers the loop immediately, and
   repeated poll-only iterations are the biggest token cost. If the only unblocked work is waiting on
   SLURM, block inside the iteration: prefer the Monitor tool with an until-loop on
   `squeue -h -j <ids>` being empty (poll every ≥ 5 min). Otherwise repeat
   `timeout 590 bash -c 'while [ -n "$(squeue -h -j <ids>)" ]; do sleep 120; done'`. Use the
   wait to do other unblocked work if there is any.
4. **Keep tool output small.** Scripts print a short summary (shapes, counts, key numbers) and write
   details to files. Use `tail -n 30` / `grep` on logs, never `cat` large files or print arrays. Use
   `pytest -q`. Don't open pickles in the session. For maps, build one low-res contact sheet
   (e.g. a 4×7 grid at ~100 dpi) and view that single image. Open an individual PNG only to
   investigate a problem.
5. **Record, tersely, before ending the iteration:**
   - `TASKS.md`: tick verified tasks with ≤ 2 lines of evidence (path, count, key number, job ID/exit
     code). Longer details go in `NOTES.md`. Never tick on assumption.
   - `## Status` (top of TASKS.md): overwrite it, ≤ 8 lines (phase, running jobs with ID and
     purpose, next step, open BLOCKED items).
   - `PROGRESS.md`: append `### Iteration N — YYYY-MM-DD HH:MM` + ≤ 4 bullets (did / verified /
     jobs / surprises). Never rewrite old entries.
   - `DECISIONS.md`: append only when you made a choice PLAN.md does not dictate that could affect
     results or reproducibility. Give date, iteration, decision, alternatives, why and effect, in ≤ 5
     lines. Changes to a pre-registered choice also go in TASKS `## Deviations`.
6. If something needs the author's judgement (indices misaligned, event R² < 0.7, > 20% coverage
   loss, C3 regression failing for unclear reasons), add a `## BLOCKED` entry in TASKS.md and move to
   other unblocked work. If *everything* is blocked on the author, update Status, then stop and
   say so plainly (do not spin).

## Guardrails
- No retraining. `strict=True` loading. No hard-coded column indices.
- Don't modify `data/drivers_data/`, `data/sensitivity_data/`, the existing driver scripts, or `paper/`. New code goes only
  in `scripts/drivers_full_legacy/`, `bashscripts/drivers_full_legacy/` and `tests/test_drivers_full_legacy.py`.
- The home quota is full. Heavy files go in `/burg-archive/glab/users/al4385/analysis/drivers_full_legacy/`. Only small
  CSVs, maps and tables go in `data/drivers_full_legacy/`.
- Pre-registered (PLAN §5.3 estimator + lag bands, §5.4 support rule, §6.4 decision rule) is fixed. Change it only
  through `## Deviations`, written *before* the affected result exists.
- SLURM: sbatch wrappers in `bashscripts/drivers_full_legacy/` with `--account=glab --partition=short`. Logs go to
  `outputs/drivers_full_legacy/logs/` (create it first). Validate on SHR first. Never resubmit PENDING/RUNNING jobs.
- No `git commit/add/push`. `rm` is aliased to `rm -i`, so use `command rm -f`, and only on files you created.

## Definition of done
Every task in TASKS Phases 0–F is `[x]` with evidence, or `[!]` with a justified BLOCKED entry that
doesn't affect the headline results. In particular: the primary CSVs exist for 6 PFTs × {SOS, EOS};
the 28 maps plus the legend exist and you checked them via the contact sheet; the occlusion
agreement table and decision-rule outcome exist per PFT × phase; the C3 regression test passed and
is recorded; `FINDINGS.md` is complete (TASKS F2) and summarises every result-affecting DECISIONS
entry; and PROGRESS.md covers every iteration. Then, and only then, output exactly:

```
<promise>DRIVERS FULL LEGACY COMPLETE</promise>
```
