#!/usr/bin/env python
"""Item 6 (PhenoCam vs model dates, step B): run TFT (Feb-2025 BDT base), PhenoFormerSeq and the 4GST OLS extrapolator
at the PhenoCam BDT pixels for 2008-2021 and compare their forecast transition dates with PhenoCam.

Windows. For every PhenoCam BDT site-year (own-PFT pixel, >= 25N) the forecast window is the FINDINGS-section-2 event window of
the observed LCSIF record at that pixel (ev2 events from lcsif_pixels.py: 30 strictly monotonic days inside the latitude
DOY band; window start = ev2 date - 15 d, +-1 d if needed to be strictly monotone). History = the 365 days before the window,
future inputs = the 30 window days, exactly as TFT_process.py builds the TFT_30 windows (scalers, feature order, soil
soil = raw class value, as stored in the TFT_30 BDT pickle; phenocam_cache.py). Model date = window start + half-amplitude crossing day of the
30-d median forecast (threshold = mean of the observed window endpoints, as score_windows.py); `t_obs` is the same
crossing of the observed LCSIF window. Because the window is anchored on the observed event this is a 30-d nowcast, not a
long-lead forecast. Years 2011-2021 are in the training years of both the TFT and PhenoFormerSeq checkpoints (in-sample);
2008-2010 are in the test split.
Validation: the window builder is checked against cached test windows of the TFT_30 pickle (inputs) and against the stored
TFT / PhenoFormerSeq predictions of those windows.
Outputs: data/review/item6/phenocam_models_{pairs,summary}_{anchor}.csv
--anchor smooth (default): the window is centred on the observed LCSIF half-amplitude date (start = date - 15 d); --anchor ev2: the
FINDINGS-2 event window (fixed DOY band: the window position itself carries almost no interannual information).
Usage: python phenocam_models.py [--years 2008 2021]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pc_models")
G = Path("/burg-archive/glab/users/al4385")
REPO = Path("/burg-archive/home/al4385/phenofusion")
SM = REPO / "data/review/item6"
sys.path.insert(0, str(REPO / "scripts/review/item6"))
sys.path.insert(0, "/burg-archive/home/al4385/code/benchmark")
from score_windows import crossing, fourgst  # noqa: E402
from PhenoFormer.model.architecture_seq import PhenoFormerSeq  # noqa: E402
from tft_torch.tft import TemporalFusionTransformer  # noqa: E402

torch.set_num_threads(4)
T0 = np.datetime64("1982-01-01")
ND = int((np.datetime64("2022-01-01") - T0).astype(int))
RAW = [
    "tmin",
    "tmax",
    "precipitation",
    "radiation",
    "photoperiod",
    "sm",
    "sif_clear_inst",
]
H, F = 365, 30


def zero_denormals(sd):
    for v in sd.values():
        v[v.abs() < 1e-30] = 0
    return sd


def load_tft(cache):
    fm, card = cache["feature_map"], cache["cardinalities"]
    cfg = {
        "optimization": {
            "batch_size": {"training": 128, "inference": 128},
            "learning_rate": 0.001,
            "max_grad_norm": 1,
        },
        "model": {
            "dropout": 0.1,
            "state_size": 160,
            "output_quantiles": [0.1, 0.5, 0.9],
            "lstm_layers": 4,
            "attention_heads": 4,
        },
        "task_type": "regression",
        "target_window_start": None,
        "data_props": {
            "num_historical_numeric": len(fm["historical_ts_numeric"]),
            "num_historical_categorical": 0,
            "num_static_numeric": len(fm["static_feats_numeric"]),
            "num_static_categorical": len(fm["static_feats_categorical"]),
            "num_future_numeric": len(fm["future_ts_numeric"]),
            "num_future_categorical": 0,
            "historical_categorical_cardinalities": [],
            "static_categorical_cardinalities": [
                card[f] + 1 for f in fm["static_feats_categorical"]
            ],
            "future_categorical_cardinalities": [],
        },
    }
    m = TemporalFusionTransformer(config=OmegaConf.create(cfg))
    sd = zero_denormals(
        torch.load(
            G
            / "weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth",
            map_location="cpu",
        )
    )
    r = m.load_state_dict(sd, strict=False)
    log.info(
        f"TFT load: missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}"
    )
    return m.eval()


def load_pf():
    m = PhenoFormerSeq(
        d_in=7,
        d_model=160,
        nhead=4,
        n_layers=4,
        dim_feedforward=320,
        dropout=0.1,
        prediction_horizon=30,
    )
    m.load_state_dict(
        zero_denormals(
            torch.load(G / "data/phenoformer/BDT_best.pth", map_location="cpu")
        )
    )
    return m.eval()


def run_tft(m, a, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(a["hist"]), bs):
            sl = slice(i, i + bs)
            b = {
                "historical_ts_numeric": torch.from_numpy(a["hist"][sl]).float(),
                "historical_ts_categorical": torch.zeros(
                    len(a["hist"][sl]), H, 0, dtype=torch.int32
                ),
                "static_feats_numeric": torch.from_numpy(a["snum"][sl]).float(),
                "static_feats_categorical": torch.from_numpy(a["scat"][sl]).int(),
                "future_ts_numeric": torch.from_numpy(a["fut"][sl]).float(),
                "future_ts_categorical": torch.zeros(
                    len(a["hist"][sl]), F, 0, dtype=torch.int32
                ),
            }
            out.append(m(b)["predicted_quantiles"][:, :, 1].numpy())
    return np.concatenate(out)


def run_pf(m, x, bs=128):
    out = []
    pos = torch.arange(H).long()
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).float()
            out.append(
                m({"climate": xb, "doys": pos[None, :].expand(len(xb), -1).clone()})[
                    "predicted_quantiles"
                ][:, :, 1].numpy()
            )
    return np.concatenate(out)


class Series:
    """Daily raw + scaled series for a set of locations (1982-01-01 .. 2021-12-31, NaN where missing)."""

    def __init__(self, locs, cache):
        sc = cache["scalers"]
        cols = ["time", "location", "latitude", "longitude", "soil"] + RAW
        tb = pq.read_table(
            G / "data/CSIFMETEO/sorted_merged_BDT_1982_2021.parquet",
            columns=cols,
            filters=[("location", "in", sorted(locs))],
        )
        log.info(f"read {tb.num_rows} rows")
        df = tb.to_pandas()
        self.raw, self.scl, self.static = {}, {}, {}
        for L, g in df.groupby("location"):
            d = (g["time"].to_numpy().astype("datetime64[D]") - T0).astype(np.int64)
            ok = (d >= 0) & (d < ND)
            raw = np.full((ND, len(RAW)), np.nan)
            raw[d[ok]] = g[RAW].to_numpy(float)[ok]
            self.raw[int(L)] = raw
            r0 = g.iloc[0]
            si = int(
                np.float64(r0["soil"]).astype(np.int32)
            )  # the TFT_30 pickle holds the raw soil class value truncated to int32 (the LabelEncoder was not applied)
            lat = sc["numeric"]["latitude"].transform([[r0["latitude"]]])[0, 0]
            lon = sc["numeric"]["longitude"].transform([[r0["longitude"]]])[0, 0]
            s = np.full_like(raw, np.nan)
            for j, c in enumerate(RAW):
                v = raw[:, j]
                f = np.isfinite(v)
                s[f, j] = sc["numeric"][c].transform(v[f].reshape(-1, 1)).ravel()
            self.scl[int(L)] = s.astype(np.float32)
            self.static[int(L)] = (
                np.array([lat, lon], np.float32),
                np.array([si], np.int32),
                float(r0["latitude"]),
            )
        del df, tb

    def window(self, L, d0):
        """History days d0-365..d0-1, forecast days d0..d0+29. None if out of range / any NaN."""
        if L not in self.scl or d0 - H < 0 or d0 + F > ND:
            return None
        s = self.scl[L][d0 - H : d0 + F]
        if not np.isfinite(s).all():
            return None
        sn, sc_, _ = self.static[L]
        return dict(hist=s[:H], fut=s[H:, :6], snum=sn, scat=sc_, target=s[H:, 6])


def stack(ws):
    return {k: np.stack([w[k] for w in ws]) for k in ws[0]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--years", type=int, nargs=2, default=[2008, 2021])
    ap.add_argument(
        "--anchor",
        choices=["ev2", "smooth"],
        default="smooth",
        help="ev2: window = FINDINGS-2 event window (fixed DOY band, so the window position carries no interannual signal); "
        "smooth: window centred (start = date - 15 d) on the observed LCSIF half-amplitude date (obs_st definition)",
    )
    a = ap.parse_args()
    cache = pickle.load(open(G / "analysis/review/item6/phenocam_cache.pkl", "rb"))
    fm = cache["feature_map"]
    assert fm["historical_ts_numeric"] == RAW and fm["future_ts_numeric"] == RAW[:6], fm
    sif_sc = cache["scalers"]["numeric"]["sif_clear_inst"]
    inv = lambda z: sif_sc.inverse_transform(
        z.reshape(-1, 1).astype(np.float64)
    ).reshape(z.shape)  # noqa: E731

    gp = pd.read_csv(SM / "ground_pixels.csv")
    gp = gp[
        (gp.source == "PhenoCam")
        & (gp.pft == "BDT")
        & gp.pft_match
        & gp.has_pixel
        & (gp.lat >= 25)
    ][["site", "location", "latitude"]].drop_duplicates("site")
    gp["location"] = gp.location.astype(int)
    log.info(f"{len(gp)} PhenoCam BDT sites, {gp.location.nunique()} pixels")
    ids = np.asarray(cache["sub"]["id"])[:, 0].astype(str)
    vloc = np.array([int(s.split("_")[0]) for s in ids])
    vday = np.array([(np.datetime64(s.split("_")[1]) - T0).astype(int) for s in ids])
    ser = Series(set(gp.location) | set(vloc), cache)
    tft, pf = load_tft(cache), load_pf()

    # ---- validation of the window builder and of the model re-runs ----
    ws = [ser.window(int(L), int(d)) for L, d in zip(vloc, vday)]
    okv = np.array([w is not None for w in ws])
    log.info(f"validation windows built: {okv.sum()}/{len(ws)}")
    b = stack([w for w in ws if w is not None])
    sub = cache["sub"]
    sel = np.flatnonzero(okv)
    chk = {
        "hist": np.abs(b["hist"] - sub["historical_ts_numeric"][sel]).max(),
        "fut": np.abs(b["fut"] - sub["future_ts_numeric"][sel]).max(),
        "snum": np.abs(b["snum"] - sub["static_feats_numeric"][sel]).max(),
        "scat": np.abs(b["scat"] - sub["static_feats_categorical"][sel]).max(),
        "target": np.abs(b["target"] - sub["target"][sel]).max(),
    }
    log.info(
        f"input check (max|built - pickle|): { {k: float(v) for k, v in chk.items()} }"
    )
    pt = run_tft(tft, b)
    pp = run_pf(pf, b["hist"])
    ct = float(np.abs(pt - cache["tft_pred"][sel][:, :, 1]).max())
    cp = (
        float(np.abs(pp - cache["pf_pred"][sel][:, :, 1]).max())
        if cache["pf_pred"].ndim == 3
        else float(np.abs(pp - cache["pf_pred"][sel]).max())
    )
    # also from the pickled arrays themselves (isolates the model from the builder)
    pt0 = run_tft(
        tft,
        dict(
            hist=sub["historical_ts_numeric"],
            snum=sub["static_feats_numeric"],
            scat=sub["static_feats_categorical"],
            fut=sub["future_ts_numeric"],
        ),
    )
    ct0 = float(np.abs(pt0 - cache["tft_pred"][:, :, 1]).max())
    log.info(
        f"max|TFT rerun - stored|: builder inputs {ct:.3g}, pickle inputs {ct0:.3g}; max|PF rerun - stored| {cp:.3g}"
    )
    with open(SM / "phenocam_models_validation.txt", "w") as fh:
        fh.write(
            f"windows built {okv.sum()}/{len(ws)}\ninput max abs diff {({k: float(v) for k, v in chk.items()})}\n"
            f"TFT max|rerun-stored| builder {ct:.3g} pickle {ct0:.3g}\nPF max|rerun-stored| {cp:.3g}\n"
        )
    if max(ct, cp) > 1e-4 or max(chk.values()) > 1e-4:
        log.error(
            "VALIDATION FAILED (>1e-4); continuing so the diagnostics are saved, do not trust the figure"
        )

    # ---- PhenoCam site-year windows ----
    ev = pd.read_parquet(G / "analysis/review/item6/pixels/BDT_ground_events.parquet")
    ev = ev[(ev["def"] == a.anchor) & ev.location.isin(gp.location)].copy()
    ev["d0"] = (
        np.floor(ev.day - 15) if a.anchor == "ev2" else np.round(ev.day - 15)
    ).astype(int)
    ev["date0"] = pd.to_datetime(T0 + ev.d0.to_numpy().astype("timedelta64[D]"))
    ev = ev[(ev.date0.dt.year >= a.years[0]) & (ev.date0.dt.year <= a.years[1])]
    ev = (
        ev.drop_duplicates(["location", "phase", "date0"])
        .assign(year=lambda d: d.date0.dt.year)
        .drop_duplicates(["location", "phase", "year"])
    )
    gs = pd.read_csv(SM / "ground_siteyears.csv")
    gs = gs[
        (gs.source == "PhenoCam") & gs.site.isin(gp.site) & gs.doy.between(30, 330)
    ].merge(gp[["site", "location"]], on="site")
    gs = gs.drop_duplicates(
        ["site", "phase", "year"]
    )  # same filters as compare_ground.py
    rows, wins = [], []
    for r in ev.itertuples():
        sgn = 1 if r.phase == "SOS" else -1
        for off in (0, -1, 1):
            w = ser.window(int(r.location), int(r.d0) + off)
            if w is None:
                continue
            y = ser.raw[int(r.location)][r.d0 + off : r.d0 + off + F, 6]
            if a.anchor == "smooth" or (sgn * np.diff(y) > 0).all():
                wins.append(w)
                rows.append((r.location, r.phase, r.d0 + off))
                break
    log.info(f"{len(ev)} ev2 events -> {len(wins)} windows with complete inputs")
    w = stack(wins)
    ptft, ppf = run_tft(tft, w), run_pf(pf, w["hist"])
    p4 = fourgst(w["hist"][:, :, 6])
    out = []
    for i, (L, ph, d0) in enumerate(rows):
        sgn = 1.0 if ph == "SOS" else -1.0
        yo = inv(w["target"][i : i + 1])
        thr = sgn * (yo[:, 0] + yo[:, -1]) / 2
        rec = dict(
            location=L,
            phase=ph,
            d0=d0,
            start=str(T0 + np.timedelta64(int(d0), "D")),
            t_obs=float(crossing(sgn * yo, thr)[0]),
        )
        for k, m in (("TFT", ptft), ("PhenoFormerSeq", ppf), ("4GST", p4)):
            t = crossing(sgn * inv(m[i : i + 1]), thr)[0]
            rec[f"t_{k}"] = float(t)  # NaN = never crosses
        out.append(rec)
    o = pd.DataFrame(out)
    o = o[
        np.isfinite(o.t_obs)
    ]  # observed window must contain the half-amplitude crossing
    o["start"] = pd.to_datetime(o.start)
    o["year"] = o.start.dt.year
    o = o.merge(
        gs[["site", "location", "phase", "year", "date", "doy"]].rename(
            columns={"doy": "doy_ground", "date": "date_ground"}
        ),
        on=["location", "phase", "year"],
    )
    o["date_ground"] = pd.to_datetime(o.date_ground)
    for k in ("obs", "TFT", "PhenoFormerSeq", "4GST"):
        dt = o.start + pd.to_timedelta(o[f"t_{k}"] * 86400, unit="s", errors="coerce")
        d = dt.dt.dayofyear.astype(float)
        d = np.where(
            dt.dt.year > o.year, d + 365, np.where(dt.dt.year < o.year, d - 365, d)
        )
        o[f"doy_{k}"] = np.where(o[f"t_{k}"].notna(), d, np.nan)
    o["period"] = np.where(o.year <= 2010, "test 2008-10", "train >=2011")
    o["site_key"] = o.site
    # reference: smooth-definition LCSIF date from the earlier pairs file
    pr = pd.read_csv(SM / "ground_vs_lcsif_pairs.csv")
    pr = pr[
        (pr.source == "PhenoCam")
        & (pr.pft == "BDT")
        & (pr["def"] == "smooth")
        & pr.pft_match
    ][["site", "phase", "year", "doy_l"]]
    o = o.merge(
        pr.rename(columns={"doy_l": "doy_smooth"}),
        on=["site", "phase", "year"],
        how="left",
    )
    o["anchor"] = a.anchor
    o.to_csv(SM / f"phenocam_models_pairs_{a.anchor}.csv", index=False)
    log.info(f"pairs: {len(o)} site-years, {o.site.nunique()} sites")

    # ---- summary ----
    def stats(d, col, label):
        d = d[np.isfinite(d[col]) & np.isfinite(d.doy_ground)]
        err = d[col] - d.doy_ground
        n = d.groupby("site").doy_ground.transform("size")
        e5 = d[n >= 5]
        ga = e5.doy_ground - e5.groupby("site").doy_ground.transform("mean")
        pa = e5[col] - e5.groupby("site")[col].transform("mean")
        r = np.corrcoef(ga, pa)[0, 1] if len(e5) > 10 and pa.std() > 0 else np.nan
        sl = np.polyfit(ga, pa, 1)[0] if len(e5) > 10 and pa.std() > 0 else np.nan
        return dict(
            model=label,
            n_siteyears=len(d),
            n_sites_ge5=int(e5.site.nunique()),
            bias_d=float(err.median()),
            mae_d=float(err.abs().mean()),
            r_within=float(r),
            slope_within=float(sl),
            sd_pred_anom=float(pa.std()),
            sd_ground_anom=float(ga.std()),
        )

    S = []
    for ph in ("SOS", "EOS"):
        d = o[o.phase == ph]
        common = d[np.isfinite(d.doy_TFT) & np.isfinite(d.doy_PhenoFormerSeq)]
        for col, lab in (
            ("doy_obs", "LCSIF window"),
            ("doy_smooth", "LCSIF smooth (ref.)"),
            ("doy_TFT", "TFT"),
            ("doy_PhenoFormerSeq", "PhenoFormerSeq"),
            ("doy_4GST", "4GST"),
        ):
            for nm, dd in (
                ("all", d),
                ("common_TFT_PF", common),
                ("test 2008-10", d[d.year <= 2010]),
                ("train >=2011", d[d.year >= 2011]),
            ):
                s = stats(dd, col, lab)
                s.update(
                    phase=ph,
                    subset=nm,
                    frac_nocross=float(dd["t_" + col[4:]].isna().mean())
                    if col[4:] in ("TFT", "PhenoFormerSeq", "4GST")
                    else 0.0,
                )
                S.append(s)
    S = pd.DataFrame(S)[
        [
            "phase",
            "subset",
            "model",
            "n_siteyears",
            "n_sites_ge5",
            "bias_d",
            "mae_d",
            "r_within",
            "slope_within",
            "sd_pred_anom",
            "sd_ground_anom",
            "frac_nocross",
        ]
    ]
    S["anchor"] = a.anchor
    S.to_csv(SM / f"phenocam_models_summary_{a.anchor}.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.subset.isin(["all", "common_TFT_PF"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

# ruff: noqa: E741, E731, F841
