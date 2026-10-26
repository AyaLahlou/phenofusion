#!/usr/bin/env python
"""Item 6: expected attenuation of the temperature sensitivity ST (d/degC) in a semi-monthly, 0.25-degree-smoothed proxy.

Synthetic pixel, 40 years. Year y: spring-mean T anomaly ~ N(0, sdT); true green-up date S_y = S0 + beta*T_y + N(0, sd_res).
Daily canopy curve = base + A*[sigmoid((t-S_y)/w) - sigmoid((t-E_y)/w_e)] (E_y = S_y + 150 + N(0,8)).
LCSIF-like observation chain (as the real product, item 2 node analysis): semi-monthly composites (mean of the curve over
days 1-15 and 16-end), + Gaussian noise (sd = noise * A) on each composite, cubic spline through the composite mid-points to
daily, then the obs_st pipeline: 31-d moving mean twice, half-amplitude up-crossing before the peak (obs_st.detect).
Scenarios vary: transition width w (days; sharp = 3, typical = 8, slow = 20), composite noise, sub-pixel heterogeneity
(pixel curve = mean of K curves whose S is offset by N(0, sd_sub) around the pixel date, same beta), and amplitude covarying with
T (A_y = A*(1 + gamma*T_y)). Recovered ST = OLS slope of detected date on T_y (per replicate, 40 y); reported: median
ratio recovered/true, its 5-95 % range over replicates, median |ST| noise (se of one pixel), and r(detected, true S).
Writes data/review/item6/sim_attenuation.csv. Usage: python sim_attenuation.py
"""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

REPO = Path("/burg-archive/home/al4385/phenofusion")
sp = importlib.util.spec_from_file_location(
    "obs_st", REPO / "scripts/ideas/idea1/obs_st.py"
)
obs = importlib.util.module_from_spec(sp)
sp.loader.exec_module(obs)
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
YEARS = 40
DAYS = np.arange(365)
EDGES = np.array(
    [
        0,
        15,
        31,
        45,
        59,
        74,
        90,
        105,
        120,
        135,
        151,
        166,
        181,
        196,
        212,
        227,
        243,
        258,
        273,
        288,
        304,
        319,
        334,
        349,
        365,
    ]
)
# semi-monthly periods (approximate, DOY edges); composite mid-points
MID = (EDGES[:-1] + EDGES[1:]) / 2


def one(rng, beta, w, noise, sub, gamma, sdT=1.5, sd_res=5.0, S0=110.0, A=1.0, K=8):
    T = rng.normal(0, sdT, YEARS)
    S = S0 + beta * T + rng.normal(0, sd_res, YEARS)
    series = []
    for y in range(YEARS):
        E = S[y] + 150 + rng.normal(0, 8)
        offs = np.linspace(-1.5, 1.5, K) * sub if sub > 0 else np.zeros(1)
        Ay = A * (1 + gamma * T[y])
        c = np.mean(
            [
                0.05
                + Ay * (sig((DAYS - S[y] - o) / w) - sig((DAYS - E - o) / max(w, 8)))
                for o in offs
            ],
            axis=0,
        )
        comp = np.array(
            [c[a:b].mean() for a, b in zip(EDGES[:-1], EDGES[1:])]
        ) + rng.normal(0, noise * A, 24)
        series.append(comp)
    comp = np.concatenate(series)
    mid = np.concatenate([MID + 365 * y for y in range(YEARS)])
    t = np.arange(365 * YEARS)
    x = CubicSpline(mid, comp)(t)
    x[: int(mid[0])] = np.nan
    x[int(mid[-1]) + 1 :] = np.nan
    xs = obs.movmean(obs.movmean(x))
    ev, _ = obs.detect(xs, np.arange(YEARS) * 365, "SOS")
    d = ev - np.arange(YEARS) * 365
    ok = np.isfinite(d)
    if ok.sum() < 15:
        return np.nan, np.nan, np.nan
    b = np.polyfit(T[ok], d[ok], 1)[0]
    return b, np.corrcoef(d[ok], S[ok])[0, 1], ok.mean()


def main():
    rng = np.random.default_rng(1)
    rows = []
    beta = -3.0
    for w, noise, sub, gamma in itertools.product(
        [3, 8, 20], [0.0, 0.05, 0.15], [0, 12], [0.0, 0.15]
    ):
        res = np.array([one(rng, beta, w, noise, sub, gamma) for _ in range(300)])
        r = res[:, 0] / beta
        rows.append(
            dict(
                width_d=w,
                noise_frac_amp=noise,
                subpixel_sd_d=sub,
                amp_covaries_gamma=gamma,
                ratio_median=np.nanmedian(r),
                ratio_p05=np.nanpercentile(r, 5),
                ratio_p95=np.nanpercentile(r, 95),
                sd_ST_estimate=np.nanstd(res[:, 0]),
                r_detected_vs_true_S=np.nanmedian(res[:, 1]),
                frac_years_detected=np.nanmedian(res[:, 2]),
            )
        )
    df = pd.DataFrame(rows)
    out = REPO / "data/review/item6/sim_attenuation.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
