#!/usr/bin/env python
"""Item 6 (benchmarks, window level): score the existing benchmark forecasts against the TFT on the SAME 30-day
test windows and the SAME event targets.

All stored benchmark predictions (code: /burg-archive/home/al4385/code/benchmark/) are (N, 30, 3) arrays aligned with
`data_sets.test` of the 2024 TFT_30 pickles (/burg-archive/glab/users/al4385/data/TFT_30/{PFT}.pickle; BDT:
merged_BDT_1982_2021.pkl), scaled target space. They carry no ids, so alignment is by row order (checked by N and by
skill). Models scored (whatever exists per PFT):
  TFT2025   /burg-archive/glab/users/al4385/predictions/pretrained_1219/pred_{PFT}_feb202025.pkl (BDT: pred_merged_BDT_feb172025.pkl)
  4GST      data/4GST/{PFT}_predictions.pkl  -- NB: FourGST_forecaster.py = OLS line through the last 30 days of SIF,
            extrapolated 30 days. It is not the 4GST algorithm; we also recompute it here to verify.
  PhenoFormerSeq  data/phenoformer/{PFT}_predictions.pkl (an adapted seq2seq PhenoFormer: 365-d history of all 7
            inputs incl. past SIF, no future weather, MSE loss, positional code = position in window, not DOY)
  LSTM      code/benchmark/results/lstm/{PFT}_lstm_predictions.pkl (if present)
  ThermalTime  data/pyphenology_benchmark/BET_ThermalTime_predictions.pkl (BET only; fitted to mean scaled CSIF as if it
            were a DOY -- scored only to document that it is not a usable benchmark)
  persistence  last observed history value held flat (reference)

Event targets (FINDINGS §2 rule applied to this split): observed 30-day target strictly increasing (SOS) / decreasing
(EOS) on every day AND start DOY inside the latitude-specific DOY band of
debugging/debugging_subset_data/sos_eos_{PFT}_merged_smoothed.json.
Timing metric (as memory_tests/battery): half-amplitude crossing day of the window, thr = (y_obs[0] + y_obs[29]) / 2 in
raw SIF units; t_obs from the observed target, t_m from the model median; censored to 0 / 30 when the model never
crosses (frac_nocross is reported).
Also: trajectory R2 (raw) on all windows and on event windows.

Output: data/review/item6/window_scores.csv (+ per-event parquet under /burg-archive/glab/users/al4385/analysis/review/item6/windows/).
Usage: python score_windows.py --pft NDT
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("score_windows")
G = Path("/burg-archive/glab/users/al4385")
REPO = Path("/burg-archive/home/al4385/phenofusion")
DATA = {p: G / f"data/TFT_30/{p}.pickle" for p in ("BET", "NET", "NDT")}
DATA["BDT"] = G / "data/TFT_30/merged_BDT_1982_2021.pkl"
TFTP = {
    p: G / f"predictions/pretrained_1219/pred_{p}_feb202025.pkl"
    for p in ("BET", "NET", "NDT")
}
TFTP["BDT"] = G / "predictions/pretrained_1219/pred_merged_BDT_feb172025.pkl"
BENCH = {
    "4GST": lambda p: G / f"data/4GST/{p}_predictions.pkl",
    "PhenoFormerSeq": lambda p: G / f"data/phenoformer/{p}_predictions.pkl",
    "LSTM": lambda p: Path(
        f"/burg-archive/home/al4385/code/benchmark/results/lstm/{p}_lstm_predictions.pkl"
    ),
    "ThermalTime": lambda p: G
    / f"data/pyphenology_benchmark/{p}_ThermalTime_predictions.pkl",
}
OUT = G / "analysis/review/item6/windows"
SMALL = REPO / "data/review/item6"


def crossing(c, thr):
    """First up-crossing (days, linear interp.) of curves c (n,T) through thr (n,); 0 if already above at day 0,
    NaN if never."""
    above = c >= thr[:, None]
    up = above[:, 1:] & ~above[:, :-1]
    has = up.any(axis=1)
    i = np.argmax(up, axis=1)
    r = np.arange(len(c))
    c0, c1 = c[r, i], c[r, i + 1]
    t = i + (thr - c0) / np.where(c1 != c0, c1 - c0, np.nan)
    t = np.where(above[:, 0], 0.0, t)
    return np.where(has | above[:, 0], t, np.nan)


def r2(y, f):
    y, f = y.ravel(), f.ravel()
    return 1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)


def load_preds(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    q = np.asarray(d["predicted_quantiles"])
    return q[:, :, 1] if q.ndim == 3 else q


def fourgst(h, look=30, H=30):
    """Re-implementation of FourGST_forecaster.predict (OLS through last `look` days, extrapolated)."""
    x = np.arange(look, dtype=float)
    y = h[:, -look:].astype(float)
    xm = x.mean()
    b = ((x - xm) * (y - y.mean(1, keepdims=True))).sum(1) / ((x - xm) ** 2).sum()
    a = y.mean(1) - b * xm
    xf = np.arange(look, look + H, dtype=float)
    return a[:, None] + b[:, None] * xf[None, :]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pft", required=True, choices=list(DATA))
    a = ap.parse_args()
    p = a.pft
    with open(DATA[p], "rb") as fh:
        d = pickle.load(fh)
    fm, sc = d["feature_map"], d["scalers"]["numeric"]
    t = d["data_sets"]["test"]
    del d
    hist_names = fm["historical_ts_numeric"]
    isif = hist_names.index("sif_clear_inst")
    ids = np.asarray(t["id"])[:, 0].astype(str)
    loc = np.array([s.split("_")[0] for s in ids]).astype(int)
    start = pd.to_datetime([s.split("_")[1] for s in ids])
    y_s = np.asarray(t["target"], dtype=np.float64)
    h_s = np.asarray(t["historical_ts_numeric"][:, :, isif], dtype=np.float64)
    lat_s = np.asarray(
        t["static_feats_numeric"][:, fm["static_feats_numeric"].index("latitude")],
        dtype=np.float64,
    )
    N = len(ids)
    log.info(
        f"{p}: N test windows {N}, locations {len(np.unique(loc))}, years {start.year.min()}-{start.year.max()}, "
        f"scalers {type(sc.get('sif_clear_inst')).__name__} lat-scaler {type(sc.get('latitude')).__name__}, hist order {hist_names}"
    )
    s_sif = sc["sif_clear_inst"]
    inv = lambda x: s_sif.inverse_transform(x.reshape(-1, 1)).reshape(x.shape)  # noqa: E731
    lat = (
        sc["latitude"].inverse_transform(lat_s.reshape(-1, 1)).ravel()
        if "latitude" in sc
        else lat_s
    )
    y = inv(y_s)
    h_last = inv(h_s[:, -30:])

    models = {}
    if TFTP[p].exists():
        models["TFT2025"] = load_preds(TFTP[p])
    for k, f in BENCH.items():
        if f(p).exists():
            models[k] = load_preds(f(p))
    re4 = fourgst(h_s)
    models["persistence"] = np.repeat(h_s[:, -1:], 30, axis=1)
    rows = []
    for k in list(models):
        m = models[k]
        if m.shape[0] != N:
            log.info(f"  {k}: N={m.shape[0]} != {N} -> not aligned, dropped")
            rows.append(
                {"pft": p, "model": k, "note": f"N mismatch {m.shape[0]} vs {N}"}
            )
            del models[k]
            continue
        log.info(f"  {k}: scaled R2 all windows {r2(y_s, m):.4f}")
    if "4GST" in models:
        log.info(
            f"  4GST stored vs re-implementation max|diff| {np.nanmax(np.abs(models['4GST'] - re4)):.3g}"
        )
    raw = {
        k: inv(m.astype(np.float64)) for k, m in models.items()
    }  # QuantileTransformer clips out-of-range values

    # FINDINGS §2 event windows on this split
    cfg = next(
        iter(
            json.load(
                open(
                    REPO
                    / f"debugging/debugging_subset_data/sos_eos_{p}_merged_smoothed.json"
                )
            ).values()
        )
    )
    keys = np.array(sorted(cfg, key=float), dtype=object)
    kv = np.array([float(x) for x in keys])
    j = np.clip(np.searchsorted(kv, lat), 0, len(kv) - 1)
    jm = np.clip(j - 1, 0, len(kv) - 1)
    j = np.where(np.abs(kv[jm] - lat) < np.abs(kv[j] - lat), jm, j)
    near = np.abs(kv[j] - lat) <= 0.13
    doy = start.dayofyear.to_numpy()
    dif = np.diff(y, axis=1)
    out_ev = []
    for ph, mono in (("SOS", (dif > 0).all(1)), ("EOS", (dif < 0).all(1))):
        band = np.zeros(N, bool)
        for i in np.flatnonzero(mono & near):
            w = cfg[keys[j[i]]][ph.lower()]
            s0, e0 = int(w["start"]), int(w["end"])
            band[i] = (s0 <= doy[i] < e0) if s0 <= e0 else (doy[i] >= s0 or doy[i] < e0)
        ev = np.flatnonzero(mono & band)
        sgn = 1.0 if ph == "SOS" else -1.0
        thr = sgn * (y[ev, 0] + y[ev, -1]) / 2
        t_obs = crossing(sgn * y[ev], thr)
        log.info(
            f"{p} {ph}: monotonic {mono.sum()}, in DOY band {len(ev)}; locations {len(np.unique(loc[ev]))}"
        )
        per_loc_years = pd.Series(start.year[ev]).groupby(loc[ev]).nunique()
        e = pd.DataFrame(
            {
                "pft": p,
                "phase": ph,
                "idx": ev,
                "location": loc[ev],
                "latitude": lat[ev],
                "start": start[ev],
                "t_obs": t_obs,
            }
        )
        for k, m in raw.items():
            tm = crossing(sgn * m[ev], thr)
            nocross = np.isnan(tm)
            tm_c = np.where(nocross, 30.0, tm)
            e[f"t_{k}"] = tm_c
            e[f"nocross_{k}"] = nocross
            err = tm_c - t_obs
            ok = np.isfinite(t_obs)
            rr = (
                np.corrcoef(tm_c[ok], t_obs[ok])[0, 1]
                if ok.sum() > 10 and np.std(tm_c[ok]) > 0
                else np.nan
            )
            rows.append(
                {
                    "pft": p,
                    "phase": ph,
                    "model": k,
                    "n_events": len(ev),
                    "n_loc": int(e.location.nunique()),
                    "n_loc_ge4_years": int((per_loc_years >= 4).sum()),
                    "bias_d": float(np.nanmean(err)),
                    "mae_d": float(np.nanmean(np.abs(err))),
                    "medae_d": float(np.nanmedian(np.abs(err))),
                    "frac_nocross": float(nocross.mean()),
                    "r_pooled": float(rr),
                    "sd_t_obs": float(np.nanstd(t_obs)),
                    "r2_event_traj_raw": float(r2(y[ev], m[ev])),
                    "r2_all_raw": float(r2(y, m)),
                    "r2_all_scaled": float(r2(y_s, models[k])),
                }
            )
        out_ev.append(e)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(out_ev).to_parquet(OUT / f"{p}_events.parquet", index=False)
    SMALL.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    f = SMALL / f"window_scores_{p}.csv"
    df.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

# ruff: noqa: E741, E731, F841
