"""Figures for review item 6 (docs/review_response/item6.md). Reads data/review/item6/*.csv."""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/review/item6")
ap.add_argument("--out-dir", default="data/review/item6/figures")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
D = lambda f: pd.read_csv(os.path.join(a.data_dir, f))

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "figure.dpi": 100,
        "savefig.dpi": 300,
    }
)
C = dict(
    tft="#1b4f9c",
    pf="#d95f02",
    lstm="#7570b3",
    gst="#66a61e",
    pers="#999999",
    tt="#e7298a",
    obs="#222222",
    sos="#2a9d5c",
    eos="#c0562b",
)
MODELS = ["TFT2025", "PhenoFormerSeq", "LSTM", "4GST", "persistence", "ThermalTime"]
MCOL = dict(zip(MODELS, [C["tft"], C["pf"], C["lstm"], C["gst"], C["pers"], C["tt"]]))
MLAB = {
    "4GST": "4GST (OLS extrap.)",
    "PhenoFormerSeq": "PhenoFormerSeq",
    "TFT2025": "TFT",
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------- Fig 1: LCSIF vs ground dates ----------
pairs = D("ground_vs_lcsif_pairs.csv")
pairs = pairs[(pairs["def"] == "smooth") & pairs.pft_match & pairs.doy_l.notna()]
cases = [("PhenoCam", "BDT", "SOS"), ("PhenoCam", "BDT", "EOS"), ("NPN", "BDT", "SOS")]
gv = D("ground_vs_lcsif.csv")
gv = gv[(gv["def"] == "smooth") & gv.pft_match].set_index(["source", "pft", "phase"])
fig, ax = plt.subplots(2, 3, figsize=(12, 7.6))
for j, (src, pft, ph) in enumerate(cases):
    d = pairs[(pairs.source == src) & (pairs.pft == pft) & (pairs.phase == ph)].copy()
    r = gv.loc[(src, pft, ph)]
    col = C["sos"] if ph == "SOS" else C["eos"]
    ttl = f"{src} {pft} {ph}"
    a0 = ax[0, j]
    a0.scatter(d.doy, d.doy_l, s=9, alpha=0.35, color=col, lw=0)
    lo, hi = np.nanmin([d.doy, d.doy_l]), np.nanmax([d.doy, d.doy_l])
    a0.plot([lo, hi], [lo, hi], "k--", lw=1)
    a0.set(xlabel="Ground date (DOY)", ylabel="LCSIF date (DOY)", title=ttl)
    a0.text(
        0.04,
        0.96,
        f"n = {int(r.n_siteyears)} site-yr\nbias {r.bias_d:+.0f} d, MAE {r.mae_d:.0f} d",
        transform=a0.transAxes,
        va="top",
        fontsize=9,
    )
    # within-site anomalies for sites with >=5 years
    d["site_key"] = d["unit"]
    n = d.groupby("site_key").doy.transform("size")
    d = d[n >= 5]
    ga = d.doy - d.groupby("site_key").doy.transform("mean")
    la = d.doy_l - d.groupby("site_key").doy_l.transform("mean")
    a1 = ax[1, j]
    a1.scatter(ga, la, s=9, alpha=0.35, color=col, lw=0)
    m = max(np.nanpercentile(np.abs(ga), 99), np.nanpercentile(np.abs(la), 99))
    a1.plot([-m, m], [-m, m], "k--", lw=1)
    a1.axhline(0, c="#bbb", lw=0.6)
    a1.axvline(0, c="#bbb", lw=0.6)
    a1.set_xlim(-m, m)
    a1.set_ylim(-m, m)
    a1.set(xlabel="Ground anomaly (d)", ylabel="LCSIF anomaly (d)")
    a1.text(
        0.04,
        0.96,
        f"within-site r = {r.r_within:.2f}\n{int(r.n_sites_ge5)} sites ≥5 y",
        transform=a1.transAxes,
        va="top",
        fontsize=9,
    )
fig.suptitle(
    "LCSIF-derived phenology dates vs ground observations (own-PFT pixels, ≥25°N)",
    fontweight="bold",
    y=1.0,
)
fig.tight_layout()
save(fig, "fig1_ground_vs_lcsif_dates")

# ---------- Fig 2: temperature sensitivity ground vs LCSIF ----------
rows = [
    ("PhenoCam", "BDT", "SOS"),
    ("PhenoCam", "BDT", "EOS"),
    ("PhenoCam", "NET", "SOS"),
    ("PhenoCam", "NET", "EOS"),
    ("NPN", "BDT", "SOS"),
    ("NPN", "BDT", "EOS"),
]
fig, (axa, axb) = plt.subplots(
    1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1])
)
y = np.arange(len(rows))[::-1]
for yi, k in zip(y, rows):
    r = gv.loc[k]
    axa.errorbar(
        r.ST_ground,
        yi + 0.15,
        xerr=[[r.ST_ground - r.ST_ground_lo], [r.ST_ground_hi - r.ST_ground]],
        fmt="o",
        color=C["obs"],
        capsize=2,
        label="Ground",
    )
    axa.errorbar(
        r.ST_lcsif,
        yi - 0.15,
        xerr=[[r.ST_lcsif - r.ST_lcsif_lo], [r.ST_lcsif_hi - r.ST_lcsif]],
        fmt="s",
        color=C["tft"],
        capsize=2,
        label="LCSIF",
    )
axa.axvline(0, c="#888", lw=0.8)
axa.set_yticks(y)
axa.set_yticklabels([f"{s} {p} {ph}" for s, p, ph in rows])
axa.set_xlabel("Temperature sensitivity ST (d/°C), 95% site-bootstrap CI")
h, l = axa.get_legend_handles_labels()
axa.legend(
    dict(zip(l, h)).values(), dict(zip(l, h)).keys(), frameon=False, loc="lower left"
)
axa.set_title("(a) ST from ground vs LCSIF dates")
rr = [
    k
    for k in rows
    if np.isfinite(gv.loc[k].ratio)
    and gv.loc[k].ratio_lo > -1
    and gv.loc[k].ratio_hi < 2
    and abs(gv.loc[k].ST_ground) > 1
]
for i, k in enumerate(rr[::-1]):
    r = gv.loc[k]
    axb.errorbar(
        r.ratio,
        i,
        xerr=[[r.ratio - r.ratio_lo], [r.ratio_hi - r.ratio]],
        fmt="o",
        color=C["tft"],
        capsize=3,
    )
axb.axvline(1, c="k", ls="--", lw=1)
axb.axvline(0, c="#888", lw=0.8)
axb.set_yticks(range(len(rr)))
axb.set_yticklabels([f"{s} {p} {ph}" for s, p, ph in rr[::-1]])
axb.set_xlabel("ST$_{LCSIF}$ / ST$_{ground}$")
axb.set_xlim(-0.4, 1.4)
axb.set_title("(b) Ratio (1 = LCSIF reproduces ground ST)")
axb.text(
    0.02,
    0.5,
    "Cases with ground ST ≈ 0 or ratio CI spanning\nunbounded values omitted (not meaningful)",
    transform=axb.transAxes,
    fontsize=8,
    color="#555",
)
fig.tight_layout()
save(fig, "fig2_sensitivity_ground_vs_lcsif")

# ---------- Fig 3: simulated attenuation ----------
s = (
    D("sim_attenuation.csv")
    .sort_values(["noise_frac_amp", "width_d", "subpixel_sd_d", "amp_covaries_gamma"])
    .reset_index(drop=True)
)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
ncol = {0.0: "#2a9d5c", 0.05: "#e9a820", 0.15: "#c0392b"}
for nz, g in s.groupby("noise_frac_amp"):
    ax[0].errorbar(
        g.index,
        g.ratio_median,
        yerr=[g.ratio_median - g.ratio_p05, g.ratio_p95 - g.ratio_median],
        fmt="o",
        ms=4,
        color=ncol[nz],
        capsize=2,
        lw=1,
        label=f"composite noise {nz*100:.0f}% of amplitude",
    )
ax[0].axhline(1, c="k", ls="--", lw=1)
ax[0].set(
    xlabel="Scenario (sorted by noise, transition width, subpixel spread, amplitude covariance)",
    ylabel="Recovered / true ST",
    ylim=(0, 2),
)
ax[0].set_title("(a) Median ratio ≈ 1 in all 36 scenarios; noise widens, not biases")
ax[0].legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
ax[1].scatter(
    s.noise_frac_amp * 100 + np.random.default_rng(0).uniform(-0.8, 0.8, len(s)),
    s.sd_ST_estimate,
    c=[ncol[n] for n in s.noise_frac_amp],
    s=25,
)
ax[1].set(
    xlabel="Composite noise (% of amplitude)",
    ylabel="s.d. of single-pixel 40-y ST (d/°C)",
)
ax[1].set_title("(b) Single-pixel precision")
fig.tight_layout()
save(fig, "fig3_sim_attenuation")

# ---------- Fig 4: benchmark timing skill ----------
W = pd.concat([D(f"window_scores_{p}.csv") for p in ["BDT", "NET", "NDT", "BET"]])
W = W[W.phase.notna()]
W["lab"] = W.pft + " " + W.phase
labs = [f"{p} {ph}" for p in ["BDT", "NET", "NDT", "BET"] for ph in ["SOS", "EOS"]]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
ms = ["TFT2025", "PhenoFormerSeq", "LSTM", "4GST", "persistence", "ThermalTime"]
bw = 0.13
for i, m in enumerate(ms):
    v = [W[(W.lab == l) & (W.model == m)].medae_d.mean() for l in labs]
    ax[0].bar(
        np.arange(len(labs)) + (i - 2.5) * bw,
        v,
        bw,
        color=MCOL[m],
        label=MLAB.get(m, m),
    )
ax[0].set_xticks(range(len(labs)))
ax[0].set_xticklabels(labs)
ax[0].set_ylabel("Median |timing error| (d)")
ax[0].set_title("(a) Transition-day error, 30-d windows (same events)")
ax[0].set_ylim(0, 20)
ax[0].legend(frameon=False, ncol=6, fontsize=8, loc="upper center", columnspacing=1)
ax[0].text(
    0.01,
    0.86,
    "BET TFT: test rows not aligned (dropped)\nThermalTime: BET only",
    transform=ax[0].transAxes,
    fontsize=8,
    color="#555",
)
learned = ["TFT2025", "PhenoFormerSeq", "LSTM"]
for i, m in enumerate(learned):
    v = [W[(W.lab == l) & (W.model == m)].r_pooled.mean() for l in labs]
    ax[1].bar(
        np.arange(len(labs)) + (i - 1) * 0.27,
        v,
        0.27,
        color=MCOL[m],
        label=MLAB.get(m, m),
    )
ax[1].set_xticks(range(len(labs)))
ax[1].set_xticklabels(labs, rotation=45, ha="right")
ax[1].set_ylabel("Pooled r (obs vs predicted timing)")
ax[1].set_title("(b) Pooled correlation (mostly spatial)")
fig.tight_layout()
save(fig, "fig4_benchmark_timing_skill")

# ---------- Fig 5: sensitivity of models vs observed ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
pf = pd.concat([D(f"pf_perturb_{p}.csv") for p in ["BDT", "NET", "NDT", "BET"]])
sc = {"warm1_last60": "+1 °C, last 60 d", "warm1_hist": "+1 °C, whole history"}
pfts = ["BDT", "NET", "NDT", "BET"]
obs = D("thermaltime_obs_st.csv").set_index("pft")
x = np.arange(len(pfts))
for i, (k, lab) in enumerate(sc.items()):
    v = [
        pf[(pf.pft == p) & (pf.phase == "SOS") & (pf.scenario == k)].dt_per_degC.iloc[0]
        for p in pfts
    ]
    ax[0].bar(
        x + (i - 0.5) * 0.3,
        v,
        0.3,
        color=[C["pf"], "#f4a261"][i],
        label=f"PhenoFormerSeq, {lab}",
    )
ax[0].scatter(
    x,
    [obs.loc[p, "ST60_obs_median"] for p in pfts],
    marker="D",
    s=55,
    color=C["obs"],
    zorder=5,
    label="Observed LCSIF ST60 (median)",
)
ax[0].scatter(
    [0],
    [-2.34],
    marker="*",
    s=170,
    color="#d4a017",
    zorder=6,
    label="PhenoCam BDT ground",
)
ax[0].axhline(0, c="#888", lw=0.8)
ax[0].set_xticks(x)
ax[0].set_xticklabels(pfts)
ax[0].set_ylabel("SOS shift (d/°C)")
ax[0].set_title("(a) PhenoFormerSeq SOS response is 5–15× too weak")
ax[0].legend(frameon=False, fontsize=8, loc="lower right")
ax[0].text(
    0.02,
    0.95,
    "4GST: ST = 0 (no weather input)\nTFT (30-d ckpts): |Δt| ≲ 1e-3 d",
    transform=ax[0].transAxes,
    va="top",
    fontsize=8,
    color="#555",
)
tp = ["BDT", "BET", "GRA", "NDT", "NET", "SHR"]
x2 = np.arange(len(tp))
ax[1].bar(
    x2 - 0.18,
    [obs.loc[p, "ST_tt_true_median"] for p in tp],
    0.36,
    color=C["tt"],
    label="Thermal-time (GDD) model, true ST",
)
ax[1].bar(
    x2 + 0.18,
    [obs.loc[p, "ST60_obs_median"] for p in tp],
    0.36,
    color=C["obs"],
    label="Observed LCSIF ST60",
)
for xi, p in zip(x2, tp):
    ax[1].text(
        xi,
        1.0,
        f"r={obs.loc[p,'r_tt_median']:.2f}",
        ha="center",
        fontsize=8,
        color="#333",
        va="top",
        transform=ax[1].get_xaxis_transform(),
    )
ax[1].axhline(0, c="#888", lw=0.8)
ax[1].set_xticks(x2)
ax[1].set_xticklabels(tp)
ax[1].set_ylabel("SOS ST (d/°C)")
ax[1].set_ylim(-6.5, 4.2)
ax[1].set_title(
    "(b) Thermal-time over-responds; wrong sign for BET/SHR\n(r = interannual agreement with LCSIF SOS)",
    fontsize=10,
)
ax[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "fig5_model_sensitivity_vs_observed")
print("ok")

# ---------- Fig 6: model vs observed transition timing, BDT events (needs pyarrow) ----------
try:
    ev = pd.read_parquet(
        "/burg-archive/glab/users/al4385/analysis/review/item6/windows/BDT_events.parquet"
    )
except Exception as ex:  # noqa: BLE001
    print("fig6 skipped:", ex)
    ev = None
if ev is not None:
    mods = [
        ("TFT2025", "TFT"),
        ("PhenoFormerSeq", "PhenoFormerSeq"),
        ("LSTM", "LSTM"),
        ("4GST", "4GST (OLS extrap.)"),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for i, ph in enumerate(["SOS", "EOS"]):
        e = ev[ev.phase == ph]
        for j, (m, lab) in enumerate(mods):
            a_ = ax[i, j]
            hb = a_.hexbin(
                e.t_obs,
                e[f"t_{m}"],
                gridsize=30,
                extent=(0, 30, 0, 30),
                mincnt=1,
                bins="log",
                cmap="viridis",
            )
            a_.plot([0, 30], [0, 30], "w--", lw=1)
            ok = ~e[f"nocross_{m}"].astype(bool)
            r = np.corrcoef(e.t_obs[ok], e[f"t_{m}"][ok])[0, 1]
            err = e[f"t_{m}"] - e.t_obs
            a_.text(
                0.04,
                0.96,
                f"r = {r:.2f}  bias {err.mean():+.1f} d\nmed|err| {err.abs().median():.1f} d  no-cross {100*(~ok).mean():.0f}%",
                transform=a_.transAxes,
                va="top",
                fontsize=8.5,
                color="w",
                bbox=dict(fc="k", alpha=0.45, lw=0),
            )
            if i == 0:
                a_.set_title(lab)
            if j == 0:
                a_.set_ylabel(f"{ph}: predicted crossing day")
            if i == 1:
                a_.set_xlabel("Observed (LCSIF) crossing day")
    fig.colorbar(hb, ax=ax, label="events (log)", shrink=0.7, pad=0.015)
    fig.suptitle(
        f"BDT 30-d forecasts, same events (2002–2010 test): predicted vs observed transition day (n={len(ev[ev.phase=='SOS']):,} SOS, {len(ev[ev.phase=='EOS']):,} EOS)",
        fontweight="bold",
    )
    save(fig, "fig6_bdt_model_vs_observed_timing")

# ---------- Fig 7: PhenoCam vs model dates at PhenoCam BDT pixels (phenocam_models.py) ----------
for anchor in ("smooth", "ev2"):
    pm_f = os.path.join(a.data_dir, f"phenocam_models_pairs_{anchor}.csv")
    if not os.path.exists(pm_f):
        continue
    pm = pd.read_csv(pm_f)
    sm_ = pd.read_csv(os.path.join(a.data_dir, f"phenocam_models_summary_{anchor}.csv"))
    sm_ = sm_[sm_.subset == "all"].set_index(["phase", "model"])
    cols = [
        ("doy_obs", "LCSIF window (observed)", "LCSIF window"),
        ("doy_TFT", "TFT", "TFT"),
        ("doy_PhenoFormerSeq", "PhenoFormerSeq", "PhenoFormerSeq"),
        ("doy_4GST", "4GST (OLS extrap.)", "4GST"),
    ]
    ccol = {
        "doy_obs": C["obs"],
        "doy_TFT": C["tft"],
        "doy_PhenoFormerSeq": C["pf"],
        "doy_4GST": C["gst"],
    }
    MK = {"test 2008-10": ("o", 1.0), "train >=2011": ("^", 0.55)}

    def pm_fig(phases, rows_kind, name, figsize):
        fig, ax = plt.subplots(len(phases) * len(rows_kind), 4, figsize=figsize)
        ax = np.atleast_2d(ax)
        ri = 0
        for ph in phases:
            for kind in rows_kind:
                for j, (c, lab, key) in enumerate(cols):
                    a_ = ax[ri, j]
                    d = pm[
                        (pm.phase == ph)
                        & np.isfinite(pm[c])
                        & np.isfinite(pm.doy_ground)
                    ].copy()
                    if kind == "anom":
                        n = d.groupby("site").doy_ground.transform("size")
                        d = d[n >= 5]
                        x = d.doy_ground - d.groupby("site").doy_ground.transform(
                            "mean"
                        )
                        y = d[c] - d.groupby("site")[c].transform("mean")
                    else:
                        x, y = d.doy_ground, d[c]
                    for per, (mk, al) in MK.items():
                        s_ = d.period == per
                        a_.scatter(
                            x[s_],
                            y[s_],
                            s=14 if mk == "o" else 12,
                            marker=mk,
                            alpha=0.55 if mk == "^" else 0.9,
                            color=ccol[c],
                            lw=0,
                            label={
                                "test 2008-10": "test period 2008–10",
                                "train >=2011": "train period ≥2011 (in-sample)",
                            }[per],
                        )
                    if len(d):
                        both = np.concatenate([x, y])
                        lo, hi = (
                            np.percentile(both, 0.5) - 5,
                            np.percentile(both, 99.5) + 5,
                        )
                    else:
                        lo, hi = 0, 1
                    if kind == "anom":
                        m = (
                            max(
                                np.percentile(np.abs(x), 99),
                                np.percentile(np.abs(y), 99),
                            )
                            * 1.05
                            if len(d)
                            else 1
                        )
                        lo, hi = -m, m
                        a_.axhline(0, c="#ccc", lw=0.6)
                        a_.axvline(0, c="#ccc", lw=0.6)
                    a_.plot([lo, hi], [lo, hi], "k--", lw=1)
                    a_.set_xlim(lo, hi)
                    a_.set_ylim(lo, hi)
                    s = sm_.loc[(ph, key)]
                    nc = (
                        f"\n{100 * s.frac_nocross:.0f}% of forecasts never cross"
                        if s.frac_nocross > 0.05
                        else ""
                    )
                    if kind == "anom" and s.n_sites_ge5 < 3:
                        a_.text(
                            0.5,
                            0.5,
                            "no site with ≥5 matched years\n(too many forecasts never cross)",
                            transform=a_.transAxes,
                            ha="center",
                            fontsize=9,
                            color="#666",
                        )
                    a_.text(
                        0.04,
                        0.96,
                        (
                            f"n = {int(s.n_siteyears)} site-yr, bias {s.bias_d:+.0f} d, MAE {s.mae_d:.0f} d{nc}"
                            if kind == "date"
                            else (
                                f"within-site r = {s.r_within:.2f}\n{int(s.n_sites_ge5)} sites ≥5 y, slope {s.slope_within:.2f}"
                                if s.n_sites_ge5 >= 3
                                else ""
                            )
                        ),
                        transform=a_.transAxes,
                        va="top",
                        fontsize=8.5,
                    )
                    if ri == 0:
                        a_.set_title(lab)
                    if kind == "date":
                        a_.set_xlabel("PhenoCam date (DOY)")
                        a_.set_ylabel(f"{ph}: model date (DOY)")
                    else:
                        a_.set_xlabel("PhenoCam anomaly (d)")
                        a_.set_ylabel(f"{ph}: model anomaly (d)")
                ri += 1
        h, l = ax[0, 0].get_legend_handles_labels()
        fig.legend(
            h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01)
        )
        fig.suptitle(
            f"PhenoCam vs model transition dates, PhenoCam BDT pixels, 30-d nowcast windows anchored on {'the observed LCSIF half-amplitude date' if anchor == 'smooth' else 'the fixed-DOY-band event window (ev2)'}",
            fontweight="bold",
            y=1.0,
            fontsize=11,
        )
        fig.text(
            0.5,
            -0.035,
            "2011–2021 lie in the training years of the TFT and PhenoFormerSeq checkpoints (in-sample); only 2008–2010 are test years. "
            "LSTM omitted (categorical embeddings not saved).",
            ha="center",
            fontsize=8.5,
            color="#444",
        )
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        save(fig, name)

    sfx = "" if anchor == "smooth" else "_ev2"
    pm_fig(
        ["SOS", "EOS"],
        ["date", "anom"],
        f"fig7_phenocam_model_anomalies{sfx}",
        (14, 14),
    )
    pm_fig(
        ["SOS", "EOS"], ["anom"], f"fig7b_phenocam_model_anomalies_only{sfx}", (14, 7)
    )

# ruff: noqa: E741, E731, F841
