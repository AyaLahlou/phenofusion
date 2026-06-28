# Reviewer Objections — `paper/nature_draft.tex`

Simulated critical peer review for a Nature-family journal (target: Nature Climate Change).
Severity: **[FATAL]** could sink the paper as-is · **[MAJOR]** must address · **[MODERATE]** should address · **[MINOR]** polish.
Each objection notes whether it is **text-fixable** (can be resolved by rewriting/reframing the manuscript) or **needs-analysis** (requires new computation/figures that live outside the .tex).

---

## A. The headline performance claim is misleading

### A1. [FATAL · text-fixable] "Predict phenological timing with R² ~ 1" conflates CSIF regression with phenology.
The abstract claims the model predicts "phenological timing with high accuracy (R² ~ 1)." But the model forecasts **daily CSIF values 30 days ahead**, and R² ≈ 0.99 is reported on the *raw CSIF time series*. A 30-day-ahead forecast of a smooth, strongly seasonal, highly autocorrelated signal is **trivially high-R²** because the seasonal cycle + persistence dominate the variance. R² ≈ 0.99 on raw CSIF says almost nothing about skill at predicting *the timing of SOS/EOS events*, which is the actual phenology problem. A reviewer will immediately flag that the headline number is the least informative metric in the paper. The honest skill number is the **anomaly R² (0.24–0.69)**, which must be the headline, with R²~1 explicitly contextualized as dominated by the seasonal cycle.

### A2. [FATAL · needs-analysis] No phenology-date error is ever reported.
The paper is about SOS/EOS *timing*, yet **not a single metric in days** (e.g., RMSE of predicted SOS date vs. observed SOS date) is reported. All metrics are on CSIF magnitude. A phenology paper at Nature must report the error in the predicted onset/senescence **date** (in days), per PFT, ideally benchmarked against process models and an observational phenology product (e.g., MODIS MCD12Q2, PhenoCam). Without this, the central claim is unsupported by the central metric.

### A3. [MAJOR · text-fixable] Anomaly R² is described as a triumph but several values are weak.
GRA = 0.24, NET = 0.33, NDT = 0.43 are modest. The text spins all of them positively. A reviewer wants an honest accounting: for which PFTs does the method genuinely beat baselines, and where does it not? Frame limitations symmetrically with strengths.

---

## B. Experimental design / data leakage

### B1. [MAJOR · text-fixable + needs-analysis] The train/test split risks temporal leakage.
Train = 1982–1992 **and** 2011–2021; validation = 1992–2002; test = 2002–2010. The test period is **temporally interpolated between two training periods**, and adjacent-year climate/CSIF autocorrelation means the test set is not independent of training. This is interpolation, not forecasting, and inflates skill. The manuscript must (a) justify why this is not leakage, and (b) ideally show a **held-out forward-in-time test** (train on past, test on strictly future years) to support any "forecasting" claim. Sliding 395-day windows with 30-day stride also create overlapping samples that can straddle the train/test boundary — confirm no window overlaps across splits.

### B2. [MAJOR · text-fixable] Samples are spatially and temporally autocorrelated; significance is overstated.
Bootstrapped 95% CIs over pixels (sensitivity analysis) treat pixels as independent, but neighboring 0.25° pixels are highly correlated. Effective sample size is far smaller than pixel count, so the "% of pixels significant" statistics overstate confidence. Need spatial-autocorrelation-aware significance (block bootstrap / spatial effective d.o.f.) or explicit caveat.

### B3. [MODERATE · text-fixable] CSIF is itself a neural-network reconstruction; circularity risk.
CSIF (Fang et al. 2024) is an NN trained on AVHRR/MODIS reflectance to emulate OCO-2 SIF. Training a second NN to predict it, then interpreting attention as "ecological drivers," risks learning the *reconstruction model's* regularities rather than biophysics. Must acknowledge and discuss this explicitly, ideally with validation against an independent observational phenology product.

---

## C. Interpretability / causal overreach

### C1. [FATAL · text-fixable] Attention weights are interpreted as causal driver importance.
The paper repeatedly treats TFT attention/variable-selection weights as "the importance of driver X" and even as evidence of "shifting climate sensitivity." Attention is **not** a causal attribution method — it reflects predictive saliency for the model's CSIF forecast, is known to be unstable/non-unique, and correlated predictors (e.g., temperature & radiation & photoperiod all covary seasonally) make per-driver weight splitting ambiguous. Every causal/sensitivity claim must be hedged to "model-attributed importance," and the limitation stated plainly. Ideally corroborate with a perturbation/ablation experiment (permute a driver, measure forecast degradation) — that *is* a causal-ish test; attention alone is not.

### C2. [MAJOR · text-fixable] The grouping of photoperiod+radiation and precip+soil-moisture is asserted, not justified.
Lumping photoperiod with radiation, and precipitation with soil moisture, can mask opposing effects (the manuscript even admits this in one line). Given that photoperiod has near-zero interannual variance, grouping it with radiation contaminates the "solar" trend. Justify the grouping or report ungrouped.

### C3. [MAJOR · text-fixable] The "shifting sensitivity" trend may be a model/data artifact, not biology.
The early epoch (1982–1991) is AVHRR-only; the late epoch (2012–2021) is MODIS-era. Δ(attribution) between epochs could reflect **changing input data quality / sensor regime** rather than real biological change. The MODIS orbit drift is mentioned only in passing. This confound must be addressed head-on; otherwise the marquee "water increasingly controls phenology" claim is attributable to data heterogeneity.

---

## D. Benchmarking / novelty

### D1. [MAJOR · needs-analysis] No quantitative benchmark against any competing model.
The paper claims superiority over process-based LSMs, ML, and prior DL "demonstrating substantial improvements," but presents **zero comparison numbers**. The repo has 4GST, PhenoFormer, and ThermalTime/pyphenology baselines. At minimum, a head-to-head table (anomaly R² and SOS/EOS-date RMSE) for BDT vs. these baselines is required to support any "advance over prior work" sentence.

### D2. [MODERATE · text-fixable] Novelty vs. Li et al. (2022) must be sharpened.
"Widespread increasing vegetation sensitivity to soil moisture" (Li et al. 2022, Nat. Comms.) already reported the water↑ signal for GPP/NDVI. The manuscript's own task notes acknowledge "Li et al. got there first on the biological signal." The text must position the novelty precisely: SOS/EOS *timing* attribution, per-PFT, globally — not the soil-moisture-sensitivity discovery itself. Currently the framing over-claims novelty.

### D3. [MODERATE · text-fixable] LSTM was dismissed in one sentence with no evidence.
"Initial attempts with LSTM yielded suboptimal results" — no numbers. Either show the LSTM comparison or remove the claim that TFT is superior to it.

---

## E. Reproducibility, statistics, and figures

### E1. [MAJOR · text-fixable] Numbers in the text are read off figure PNGs, not reproduced.
Per `tasks.md`, anomaly R² were "read from figure images." Reviewers require a metrics table with exact values, n per PFT, and uncertainty. Add a proper Table with all PFTs × {overall, across-site, seasonal, anomaly} × {R², RMSE, bias} and sample sizes.

### E2. [MODERATE · text-fixable] No uncertainty on predictions despite a quantile model.
The TFT outputs 0.1/0.5/0.9 quantiles, but the paper never reports calibration or prediction intervals. For an ecological-forecasting paper, quantile calibration (coverage of the 80% interval) is expected.

### E3. [MINOR · text-fixable] Citation style is inconsistent.
Mixed `\citep{}`, `\cite{}`, and hard-coded "(Richardson et al., 2012)", "(Smith et al., 2010)", "(Basler, 2016)", "(Jolly & Running, 2004)", "(Chuine, 2000)" that are **not in references.bib** and won't compile as citations. All must be converted to bib keys and added to `references.bib`.

### E4. [MINOR · text-fixable] Figure cross-references are broken/placeholder.
"Figures Y1–Y4" referenced in the Leaf-out section do not exist. SVG vs PNG figure mixing; `driver_map.svg` predates GRA/SHR per tasks.md and won't match the 6-PFT text. Several figures referenced by relative path (`../data/...`) won't survive submission packaging.

### E5. [MINOR · text-fixable] Typos and prose issues.
"Plant phenology,the timing" (missing space); curly vs straight apostrophes ("model's", "deep learning's"); "suggest many regions" (grammar); duplicated limitations across Discussion and Conclusion.

---

## F. Framing / scope

### F1. [MODERATE · text-fixable] Abstract over-promises ESM integration.
The intro/abstract repeatedly promise "actionable insights for improving land surface models," but the paper delivers no LSM experiment or parameterization. Either temper to "implications for" or add a concrete LSM-relevant deliverable.

### F2. [MODERATE · text-fixable] "Memory effects" interpretation needs a guardrail.
Attention peaking ~338 days prior is interpreted as "12-month ecological memory." But with a 365-day input window and a strongly annual signal, the model attending to "same calendar period last year" can be a **trivial seasonal-phase lock**, not ecological memory. Distinguish genuine legacy from seasonal periodicity (e.g., does attention at t-338 add skill beyond the seasonal climatology?).

### F3. [MINOR · text-fixable] Title "Blooming Insights" is informal for the venue.
Consider a precise, declarative title.

---

## Priority queue for fixing (this Ralph loop)
1. **C1 + A1** — reframe attention→"model-attributed importance"; make anomaly R² the headline, contextualize R²~1. (text)
2. **E3 + E4 + E5** — fix broken citations, placeholder figure refs, typos so the document compiles cleanly. (text)
3. **D2 + D3 + F1** — sharpen novelty vs. Li et al., temper LSTM/ESM over-claims. (text)
4. **B1 + B2 + C3** — add honest leakage/autocorrelation/sensor-regime caveats to Methods + Limitations. (text)
5. **A3 + F2** — balanced reporting of weak PFTs; guardrail on memory interpretation. (text)
6. **E1 + E2** — add metrics table + quantile calibration mention. (text scaffold; numbers need analysis)
7. **A2 + D1 + B-analysis** — SOS/EOS-date errors and benchmark table. (needs-analysis — flag for author, scaffold placeholders)

Items marked needs-analysis (A2, D1, partial B1/E1/E2) cannot be fully resolved from the .tex alone and are flagged for the authors with clearly-marked placeholders rather than fabricated numbers.
