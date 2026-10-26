#!/usr/bin/env python
"""Item 6: ground (PhenoCam, USA-NPN) vs LCSIF-derived SOS/EOS: date bias, interannual r, temperature sensitivity.

Inputs
  data/review/item6/ground_siteyears.csv, ground_pixels.csv      (ground_build.py)
  analysis/review/item6/pixels/{PFT}_ground_{events.parquet,T.npz} (lcsif_pixels.py; defs smooth / node / ev2)
Matching
  site -> 0.25 deg pixel of the site's own PFT dataset (pft_match=True, primary) or any PFT dataset at that grid point
  (pft_match=False, sensitivity). Ground year = calendar year of the ground date; LCSIF event = the event of the same phase
  whose calendar year is the same (first if several). Only |lat| >= 25 N (NH) sites, years 1982-2021 (LCSIF record),
  doy of ground date in [30, 330].
Metrics per source x PFT x phase x def (all on the identical matched site-years)
  n, n_sites, bias (LCSIF - ground, days, circular), MAE, r_pooled, r_within (site-demeaned = interannual), sd_ground,
  sd_lcsif (within-site interannual sd).
  ST (d/degC): pooled within-site OLS slope of date on mean pixel T (ERA5 tmin/tmax mean) over the 60 days ending at the
  site's mean GROUND date (same window and same T for both dates), sites with >= 5 matched years. Site-bootstrap
  (B=1000) 95 % CI of ST_ground, ST_lcsif and of the ratio ST_lcsif / ST_ground (attenuation factor).
Outputs: data/review/item6/ground_vs_lcsif.csv, data/review/item6/ground_vs_lcsif_pairs.csv (site-years).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/burg-archive/home/al4385/phenofusion")
SMALL = REPO / "data/review/item6"
PIX = Path("/burg-archive/glab/users/al4385/analysis/review/item6/pixels")
T0 = np.datetime64("1982-01-01")
W = 60
MINY = 5
B = 1000


def wrap(d):
    return (d + 182.5) % 365 - 182.5


def demean(df, col, by="unit"):
    return df[col] - df.groupby(by)[col].transform("mean")


def pooled_slope(y, x):
    d = (x * x).sum()
    return (x * y).sum() / d if d > 0 else np.nan


def boot(units, yg, yl, x, rng):
    """units: array of unit labels per row (already demeaned arrays yg, yl, x). Returns draws of (stg, stl, ratio)."""
    u, inv = np.unique(units, return_inverse=True)
    sxy_g = np.bincount(inv, x * yg)
    sxy_l = np.bincount(inv, x * yl)
    sxx = np.bincount(inv, x * x)
    out = []
    for _ in range(B):
        k = rng.integers(0, len(u), len(u))
        d = sxx[k].sum()
        if d <= 0:
            continue
        g, l = sxy_g[k].sum() / d, sxy_l[k].sum() / d
        out.append((g, l, l / g if abs(g) > 0.05 else np.nan))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tag", default="ground")
    a = ap.parse_args()
    gy = pd.read_csv(SMALL / "ground_siteyears.csv")
    gp = pd.read_csv(SMALL / "ground_pixels.csv")
    gp = gp[gp.has_pixel].copy()
    gp["location"] = gp.location.astype(int)
    gy = gy[(gy.lat >= 25) & gy.doy.between(30, 330) & gy.year.between(1982, 2021)]
    rows, pairs = [], []
    rng = np.random.default_rng(0)
    for pp in sorted(gp.pix_pft.unique()):
        ev = pd.read_parquet(PIX / f"{pp}_{a.tag}_events.parquet")
        z = np.load(PIX / f"{pp}_{a.tag}_T.npz")
        locs = list(z["locations"])
        T = z["T"]
        jan1 = z["jan1"]
        li = {l: i for i, l in enumerate(locs)}
        d = T0 + np.floor(ev.day.to_numpy()).astype(int).astype("timedelta64[D]")
        ev["year"] = d.astype("datetime64[Y]").astype(int) + 1970
        ev["doy"] = (d - d.astype("datetime64[Y]")).astype(int) + 1
        ev = ev.drop_duplicates(["location", "phase", "def", "year"])
        m = gp[gp.pix_pft == pp][
            ["source", "site", "pft", "pix_pft", "location", "pft_match"]
        ]
        g = gy.merge(m, on=["source", "site", "pft"])
        g = g.merge(
            ev[["location", "phase", "def", "year", "doy"]].rename(
                columns={"doy": "doy_l"}
            ),
            on=["location", "phase", "year"],
        )
        g["diff"] = wrap(g.doy_l - g.doy)
        g["unit"] = g.source + ":" + g.site + ":" + g.pft + ":" + g.pix_pft
        # temperature window: 60 d ending at the site's mean ground date (per unit x phase)
        mg = g.groupby(["unit", "phase"]).doy.transform("mean").round().astype(int)
        end = jan1[g.year.to_numpy() - 1982] + mg.to_numpy() - 1
        tl = np.array([li[l] for l in g.location])
        win = np.array(
            [
                np.nanmean(T[i, e - W : e]) if e - W >= 0 else np.nan
                for i, e in zip(tl, end)
            ]
        )
        g["Tpre"] = win
        pairs.append(g)
    P = pd.concat(pairs, ignore_index=True)
    P.to_csv(SMALL / "ground_vs_lcsif_pairs.csv", index=False)
    for (src, pft, ph, df_, pm), g in P.groupby(
        ["source", "pft", "phase", "def", "pft_match"]
    ):
        g = g.dropna(subset=["Tpre"])
        n_all = len(g)
        cnt = g.groupby("unit").year.transform("count")
        gg = g[cnt >= MINY].copy()
        r = {
            "source": src,
            "pft": pft,
            "phase": ph,
            "def": df_,
            "pft_match": bool(pm),
            "n_siteyears": n_all,
            "n_sites": g.unit.nunique(),
            "n_sites_ge5": gg.unit.nunique(),
            "n_siteyears_ge5": len(gg),
            "bias_d": float(g["diff"].median()),
            "mean_bias_d": float(g["diff"].mean()),
            "mae_d": float(g["diff"].abs().mean()),
            "medae_d": float(g["diff"].abs().median()),
        }
        if len(g) > 5:
            r["r_pooled"] = float(np.corrcoef(g.doy, g.doy_l)[0, 1])
        if len(gg) >= 20 and gg.unit.nunique() >= 4:
            yg, yl, x = (
                demean(gg, "doy").to_numpy(),
                demean(gg, "doy_l").to_numpy(),
                demean(gg, "Tpre").to_numpy(),
            )
            # circular safety: within-site deviations beyond 100 d are errors in matching, drop
            ok = (np.abs(yg) < 60) & (np.abs(yl) < 60)
            gg2, yg, yl, x = gg[ok], yg[ok], yl[ok], x[ok]
            r["r_within"] = float(np.corrcoef(yg, yl)[0, 1])
            r["sd_within_ground"] = float(yg.std())
            r["sd_within_lcsif"] = float(yl.std())
            r["sd_Tpre_within"] = float(x.std())
            r["ST_ground"] = float(pooled_slope(yg, x))
            r["ST_lcsif"] = float(pooled_slope(yl, x))
            r["r_T_ground"] = float(np.corrcoef(yg, x)[0, 1])
            r["r_T_lcsif"] = float(np.corrcoef(yl, x)[0, 1])
            bs = boot(gg2.unit.to_numpy(), yg, yl, x, rng)
            for k, nm in enumerate(("ST_ground", "ST_lcsif", "ratio")):
                r[nm + "_lo"], r[nm + "_hi"] = (
                    float(np.nanquantile(bs[:, k], 0.025)),
                    float(np.nanquantile(bs[:, k], 0.975)),
                )
            r["ratio"] = (
                r["ST_lcsif"] / r["ST_ground"] if abs(r["ST_ground"]) > 0.05 else np.nan
            )
            # bias-corrected for the error in the LCSIF date that is uncorrelated with T: nothing; report r only.
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(SMALL / "ground_vs_lcsif.csv", index=False)
    pd.set_option("display.width", 300, "display.max_columns", 40)
    cols = [
        "source",
        "pft",
        "phase",
        "def",
        "pft_match",
        "n_siteyears",
        "n_sites",
        "n_sites_ge5",
        "bias_d",
        "mae_d",
        "r_pooled",
        "r_within",
        "sd_within_ground",
        "sd_within_lcsif",
        "ST_ground",
        "ST_lcsif",
        "ratio",
        "ratio_lo",
        "ratio_hi",
    ]
    print(R[R.pft_match][cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

# ruff: noqa: E741, E731, F841
