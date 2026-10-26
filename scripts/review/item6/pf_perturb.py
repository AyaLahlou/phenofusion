#!/usr/bin/env python
"""Item 6: does the PhenoFormerSeq benchmark respond to warming? (counterpart of the TFT history-warming battery)

PhenoFormerSeq (code: /burg-archive/home/al4385/code/benchmark/PhenoFormer/model/architecture_seq.py, weights
data/phenoformer/{PFT}_best.pth, d_model 160, 4 heads, 4 layers, ff 320) maps the 365-d history of all 7 inputs (incl. past SIF)
to the 30-d SIF forecast. It has no future-weather input. For a seeded subsample of event windows (windows parquet from
score_windows.py) we add +1 degC / +4 degC to tmin and tmax (raw space via the PFT scalers) over (a) the last 60 history days,
(b) the whole history, and report the shift of the half-amplitude crossing of the 30-d median forecast (days, + = later),
the same dt definition as the TFT battery. The unperturbed forecast is first checked against the stored predictions pickle.
4GST is a function of past SIF only (ST = 0 identically) and needs no run.
Output: data/review/item6/pf_perturb_{PFT}.csv. Usage: python pf_perturb.py --pft NDT [--n 3000]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pf")
G = Path("/burg-archive/glab/users/al4385")
sys.path.insert(0, "/burg-archive/home/al4385/phenofusion/scripts/review/item6")
sys.path.insert(0, "/burg-archive/home/al4385/code/benchmark")
from score_windows import DATA, crossing  # noqa: E402
from PhenoFormer.model.architecture_seq import PhenoFormerSeq  # noqa: E402

torch.set_num_threads(2)


def predict(model, x, bs=128):
    out = []
    pos = torch.arange(x.shape[1]).long()
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).float()
            out.append(
                model(
                    {"climate": xb, "doys": pos[None, :].expand(len(xb), -1).clone()}
                )["predicted_quantiles"][:, :, 1].numpy()
            )
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pft", required=True)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    p = a.pft
    ev = pd.read_parquet(G / f"analysis/review/item6/windows/{p}_events.parquet")
    with open(DATA[p], "rb") as fh:
        d = pickle.load(fh)
    fm, sc, t = d["feature_map"], d["scalers"]["numeric"], d["data_sets"]["test"]
    names = fm["historical_ts_numeric"]
    H = t["historical_ts_numeric"]
    del d
    model = PhenoFormerSeq(
        d_in=len(names),
        d_model=160,
        nhead=4,
        n_layers=4,
        dim_feedforward=320,
        dropout=0.1,
        prediction_horizon=30,
    )
    sd = torch.load(G / f"data/phenoformer/{p}_best.pth", map_location="cpu")
    nd = 0
    for k, v in sd.items():
        m = v.abs() < 1e-30
        nd += int((m & (v != 0)).sum())
        v[m] = 0
    model.load_state_dict(sd)
    model.eval()
    log.info(f"{p}: loaded, zeroed {nd} denormals")
    s_sif = sc["sif_clear_inst"]
    rng = np.random.default_rng(0)
    rows = []
    stored = None
    for ph in ("SOS", "EOS"):
        e = ev[ev.phase == ph]
        e = e.iloc[rng.permutation(len(e))[: a.n]].sort_values("idx")
        idx = e.idx.to_numpy()
        x0 = np.asarray(H[idx], dtype=np.float32)
        base = predict(model, x0)
        if stored is None:
            with open(G / f"data/phenoformer/{p}_predictions.pkl", "rb") as fh:
                stored = pickle.load(fh)["predicted_quantiles"][:, :, 1]
        chk = float(np.abs(base - stored[idx]).max())
        log.info(f"{ph}: n={len(idx)}  max|recomputed - stored| = {chk:.3g}")
        sgn = 1.0 if ph == "SOS" else -1.0
        raw = lambda z: s_sif.inverse_transform(
            z.reshape(-1, 1).astype(np.float64)
        ).reshape(z.shape)  # noqa: E731
        yb = raw(base)
        thr = (
            sgn * (yb[:, 0] + yb[:, -1]) / 2
        )  # note: threshold from the base forecast's own endpoints
        tb = crossing(sgn * yb, thr)
        for name, dT, span in (
            ("warm1_last60", 1.0, 60),
            ("warm1_hist", 1.0, 365),
            ("warm4_hist", 4.0, 365),
        ):
            x1 = x0.copy()
            for v in ("tmin", "tmax"):
                j = names.index(v)
                col = x0[:, 365 - span :, j]
                r = sc[v].inverse_transform(col.reshape(-1, 1).astype(np.float64)) + dT
                x1[:, 365 - span :, j] = sc[v].transform(r).reshape(col.shape)
            y1 = raw(predict(model, x1))
            t1 = crossing(sgn * y1, thr)
            dt = t1 - tb
            ok = np.isfinite(dt)
            rows.append(
                dict(
                    pft=p,
                    phase=ph,
                    scenario=name,
                    n=len(idx),
                    n_defined=int(ok.sum()),
                    dT=dT,
                    dt_mean_d=float(np.nanmean(dt)),
                    dt_median_d=float(np.nanmedian(dt)),
                    dt_per_degC=float(np.nanmean(dt) / dT),
                    mean_abs_forecast_change_scaled=float(
                        np.abs(predict(model, x1) - base).mean()
                    ),
                    check_max_abs_vs_stored=chk,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(
        Path("/burg-archive/home/al4385/phenofusion/data/review/item6")
        / f"pf_perturb_{p}.csv",
        index=False,
    )
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

# ruff: noqa: E741, E731, F841
