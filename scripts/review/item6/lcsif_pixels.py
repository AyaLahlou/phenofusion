#!/usr/bin/env python
"""Item 6: LCSIF-derived SOS/EOS dates (three definitions) + daily ERA5 temperature for a set of pixels.

Input: daily parquets (CSIFMETEO_smfix where it exists, else CSIFMETEO; CSIF and temperature are identical in both).
Locations: --locs (comma list) or --locs-file (one id per line) and/or --sample N (seeded random seasonal pixels from
the idea-1 obs_st pixel table).

Date definitions, per season-year (season-year = from the pixel's climatological trough, as obs_st.py):
  smooth : obs_st.py definition -- 31-d moving mean applied twice, half-amplitude crossing (SOS = last up-crossing
           before the peak, EOS = first down-crossing after it), linear interpolation.
  node   : the same crossing rule on the NATIVE nodes only. The daily record is piecewise linear/quadratic between nodes on
           the 1st and 15th of each month (find_nodes2.py); we keep those days, interpolate linearly between them and
           apply no smoothing.
  ev2    : FINDINGS §2 event rule on the daily record: a 30-day window whose CSIF rises (SOS) / falls (EOS) strictly on
           every day and whose start DOY is inside the latitude band of sos_eos_{PFT}_merged_smoothed.json. Date =
           median start of the first run of qualifying windows + 15 d (window centre). 30-d granularity by design.
Output (/burg-archive/glab/users/al4385/analysis/review/item6/pixels/):
  {PFT}_{tag}_events.parquet  location, latitude, longitude, phase, season_year, def, day (absolute day since 1982-01-01)
  {PFT}_{tag}_T.npz           locations, T (n_loc, n_days) daily mean temperature °C, jan1 index, anchors
Usage: python lcsif_pixels.py --pft BDT --locs-file f.txt --tag ground [--sample 2000]
"""

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("lcsif_pixels")
REPO = Path("/burg-archive/home/al4385/phenofusion")
_spec = importlib.util.spec_from_file_location(
    "obs_st", REPO / "scripts/ideas/idea1/obs_st.py"
)
obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obs)  # read-only reuse of the idea-1 detection code

G = Path("/burg-archive/glab/users/al4385")
FIX = G / "data/CSIFMETEO_smfix"
FILES = {
    p: (FIX / f.relative_to(obs.SRC) if (FIX / f.relative_to(obs.SRC)).exists() else f)
    for p, f in obs.FILES.items()
}
OUT = G / "analysis/review/item6/pixels"
PIX = G / "analysis/ideas/idea1/obs_st/all_pixels.parquet"
T0, ND, YEARS = obs.T0, obs.ND, obs.YEARS


def node_series(x, days):
    """Values on the 1st and 15th of each month, linearly interpolated back to daily."""
    dom = (days - days.astype("datetime64[M]")).astype(int) + 1
    k = np.flatnonzero(((dom == 1) | (dom == 15)) & np.isfinite(x))
    if len(k) < 10:
        return np.full_like(x, np.nan)
    out = np.interp(np.arange(len(x)), k, x[k])
    out[: k[0]] = np.nan
    out[k[-1] + 1 :] = np.nan
    return out


def ev2_dates(x, anchors, band, doy_of_day):
    """FINDINGS §2 events per season-year. band = dict phase -> (start, end) DOY."""
    res = {}
    dif = np.diff(x)
    for ph, sgn in (("SOS", 1), ("EOS", -1)):
        good = np.isfinite(dif) & (sgn * dif > 0)
        # run[i] = True if days i..i+29 strictly monotonic (29 consecutive positive diffs starting at i)
        c = np.concatenate([[0], np.cumsum(good)])
        n = len(x) - 30
        mono = np.zeros(len(x), bool)
        mono[: n + 1] = (c[29 : 29 + n + 1] - c[: n + 1]) == 29
        s0, e0 = band[ph]
        d = doy_of_day
        inb = (d >= s0) & (d < e0) if s0 <= e0 else ((d >= s0) | (d < e0))
        q = mono & inb
        out = np.full(len(anchors), np.nan)
        for i, a in enumerate(anchors):
            if a < 0 or a + 365 > len(x):
                continue
            w = np.flatnonzero(q[a : a + 365])
            if not len(w):
                continue
            brk = np.flatnonzero(np.diff(w) > 1)
            run = w[: brk[0] + 1] if len(brk) else w
            out[i] = a + np.median(run) + 15
        res[ph] = out
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pft", required=True, choices=list(FILES))
    ap.add_argument("--locs", default="")
    ap.add_argument("--locs-file", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    t0 = time.time()
    locs = set()
    if a.locs:
        locs |= {int(v) for v in a.locs.split(",")}
    if a.locs_file:
        locs |= {int(float(v)) for v in open(a.locs_file).read().split()}
    if a.sample:
        px = pd.read_parquet(PIX, columns=["location", "pft", "phase", "seasonal"])
        px = px[(px.pft == a.pft) & (px.phase == "SOS") & px.seasonal.astype(bool)]
        locs |= set(
            px.location.sample(min(a.sample, len(px)), random_state=0).astype(int)
        )
    locs = sorted(locs)
    f = FILES[a.pft]
    log.info(f"{a.pft}: {len(locs)} locations from {f}")
    cols = [
        "time",
        "location",
        "sif_clear_inst",
        "tmin",
        "tmax",
        "latitude",
        "longitude",
    ]
    tb = pq.read_table(f, columns=cols, filters=[("location", "in", locs)])
    log.info(f"read {tb.num_rows} rows in {time.time() - t0:.0f}s")
    loc = tb["location"].to_numpy()
    o = np.lexsort((tb["time"].to_numpy(), loc))
    loc = loc[o]
    day = ((tb["time"].to_numpy().astype("datetime64[D]") - T0).astype(np.int64))[o]
    arr = {c: tb[c].to_numpy().astype(np.float64)[o] for c in cols[2:]}
    del tb
    jan1 = np.array([(np.datetime64(f"{y}-01-01") - T0).astype(int) for y in YEARS])
    days = T0 + np.arange(ND).astype("timedelta64[D]")
    doy_of_day = (days - days.astype("datetime64[Y]")).astype(int) + 1
    cfg = next(
        iter(
            json.load(
                open(
                    REPO
                    / f"debugging/debugging_subset_data/sos_eos_{a.pft}_merged_smoothed.json"
                )
            ).values()
        )
    )
    kv = np.array(sorted(float(k) for k in cfg))
    kstr = {float(k): k for k in cfg}
    ul, st = np.unique(loc, return_index=True)
    en = np.append(st[1:], len(loc))
    rows, Ts, anch = [], [], []
    for n, (L, s, e) in enumerate(zip(ul, st, en)):
        d = day[s:e]
        ok = (d >= 0) & (d < ND)
        full = {}
        for c in ("sif_clear_inst", "tmin", "tmax"):
            z = np.full(ND, np.nan)
            z[d[ok]] = arr[c][s:e][ok]
            full[c] = z
        x = full["sif_clear_inst"]
        T = (full["tmin"] + full["tmax"]) / 2 - 273.15
        lat, lon = arr["latitude"][s], arr["longitude"][s]
        xs = obs.movmean(obs.movmean(x))
        doy = np.minimum(
            (
                np.arange(ND)
                - jan1[np.searchsorted(jan1, np.arange(ND), side="right") - 1]
            ),
            364,
        )
        cnt = np.bincount(doy, weights=np.isfinite(xs), minlength=365)
        sm = np.bincount(doy, weights=np.where(np.isfinite(xs), xs, 0), minlength=365)
        clim = obs.circ_smooth(sm / np.maximum(cnt, 1))
        trough = int(np.nanargmin(clim))
        anchors = jan1 + trough
        xn = node_series(x, days)
        kk = kv[np.argmin(np.abs(kv - lat))]
        band = {
            ph: (
                int(cfg[kstr[kk]][ph.lower()]["start"]),
                int(cfg[kstr[kk]][ph.lower()]["end"]),
            )
            for ph in ("SOS", "EOS")
        }
        e2 = ev2_dates(x, anchors, band, doy_of_day)
        for ph in ("SOS", "EOS"):
            for name, series in (("smooth", xs), ("node", xn)):
                ev, _ = obs.detect(series, anchors, ph)
                for i in np.flatnonzero(np.isfinite(ev)):
                    rows.append(
                        (int(L), lat, lon, ph, int(YEARS[i]), name, float(ev[i]))
                    )
            for i in np.flatnonzero(np.isfinite(e2[ph])):
                rows.append(
                    (int(L), lat, lon, ph, int(YEARS[i]), "ev2", float(e2[ph][i]))
                )
        Ts.append(T.astype(np.float32))
        anch.append(anchors)
        if n % 200 == 0:
            log.info(f"  {n}/{len(ul)} ({time.time() - t0:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.DataFrame(
        rows,
        columns=[
            "location",
            "latitude",
            "longitude",
            "phase",
            "season_year",
            "def",
            "day",
        ],
    )
    ev["pft"] = a.pft
    ev.to_parquet(OUT / f"{a.pft}_{a.tag}_events.parquet", index=False)
    np.savez_compressed(
        OUT / f"{a.pft}_{a.tag}_T.npz",
        locations=ul,
        T=np.array(Ts),
        jan1=jan1,
        anchors=np.array(anch),
    )
    log.info(f"wrote {len(ev)} events for {len(ul)} locations; {time.time() - t0:.0f}s")
    print(ev.groupby(["phase", "def"]).size().to_string())


if __name__ == "__main__":
    main()
