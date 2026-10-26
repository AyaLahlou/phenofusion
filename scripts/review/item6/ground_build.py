#!/usr/bin/env python
"""Item 6: build ground phenology site-year tables (PhenoCam, USA-NPN) and match them to 0.25° model pixels.

PhenoCam: per ROI, `gcc_90` transition_50 dates (50 % of the GCC amplitude; rising = SOS, falling = EOS). Several
cycles per year are possible (grass, shrub, agriculture-like ROIs): per ROI x direction x calendar year we keep the
transition of the largest-amplitude cycle (max_gcc - min_gcc). One ROI per site and veg type (lowest ROI number).
USA-NPN: site-level first-yes DOY per site x species x phenophase x year; only records with a prior "no" within 14 days
(0 < mean_numdays_since_prior_no <= 14, the usual NPN quality filter). SOS = onset phenophases (breaking leaf buds,
initial growth, emerging needles/leaves), EOS = colored leaves / needles (falling leaves only where no colored record).
Per site x PFT x year: median over species.
Veg type -> PFT: PhenoCam DB->BDT, EN->NET, DN->NDT, GR->GRA, SH->SHR, EB->BET; NPN Deciduous broadleaf->BDT,
Evergreen conifer/Pine->NET, Deciduous conifer->NDT, Graminoid->GRA, Drought deciduous broadleaf->SHR,
Evergreen broadleaf->BET.
Pixel match: nearest 0.25° grid point (the model pixels sit on multiples of 0.25°). A site is matched to the pixel of
its own PFT's dataset if that dataset contains the grid point ("pft_match"); otherwise to any PFT dataset that does
(kept but flagged, pft_match=False). Pixel tables from the idea-1 obs_st run (all_pixels.parquet).

Outputs (data/review/item6/): ground_siteyears.csv (source, site, pft, lat, lon, phase, year, doy, n_species),
ground_pixels.csv (site -> location per PFT dataset). Usage: python ground_build.py
"""

import glob
import json
from pathlib import Path

import pandas as pd

G = Path("/burg-archive/glab/users/al4385/data/review/item6/ground")
PIX = Path(
    "/burg-archive/glab/users/al4385/analysis/ideas/idea1/obs_st/all_pixels.parquet"
)
SMALL = Path("/burg-archive/home/al4385/phenofusion/data/review/item6")
PC2PFT = {"DB": "BDT", "EN": "NET", "DN": "NDT", "GR": "GRA", "SH": "SHR", "EB": "BET"}
NPN2PFT = {
    "Deciduous broadleaf": "BDT",
    "Evergreen conifer": "NET",
    "Pine": "NET",
    "Deciduous conifer": "NDT",
    "Graminoid": "GRA",
    "Drought deciduous broadleaf": "SHR",
    "Evergreen broadleaf": "BET",
}
NPN_SOS = {371, 373, 482, 492, 508, 496, 232}
NPN_EOS_COL = {498, 499}
NPN_EOS_FALL = {471}


def phenocam():
    rois = pd.DataFrame(json.load(open(G / "phenocam/roilists.json")))
    rois = rois[rois.roitype.isin(PC2PFT)]
    rois = rois.sort_values("sequence_number").drop_duplicates(["site", "roitype"])
    rows = []
    for _, r in rois.iterrows():
        f = G / f"phenocam/transitions/{r.roi_name}_3day_transition_dates.csv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        try:
            t = pd.read_csv(f, comment="#")
        except Exception:  # noqa: BLE001
            continue
        if "gcc_value" not in t:
            continue
        t = t[t.gcc_value == "gcc_90"].copy()
        if t.empty:
            continue
        t["date"] = pd.to_datetime(t.transition_50, errors="coerce")
        t = t.dropna(subset=["date"])
        t["amp"] = t.max_gcc - t.min_gcc
        t["year"] = t.date.dt.year
        t = t.sort_values("amp", ascending=False).drop_duplicates(["direction", "year"])
        for _, x in t.iterrows():
            rows.append(
                {
                    "source": "PhenoCam",
                    "site": r.site,
                    "veg": r.roitype,
                    "pft": PC2PFT[r.roitype],
                    "lat": r.lat,
                    "lon": r.lon,
                    "phase": "SOS" if x.direction == "rising" else "EOS",
                    "year": int(x.year),
                    "date": x.date,
                    "doy": int(x.date.dayofyear),
                    "n_species": 1,
                    "amp_gcc": float(x.amp),
                }
            )
    return pd.DataFrame(rows)


def npn():
    sp = pd.DataFrame(json.load(open(G / "npn/species.json")))[
        ["species_id", "functional_type"]
    ]
    recs = []
    for f in sorted(glob.glob(str(G / "npn/sitelevel_*.json"))):
        d = json.load(open(f))
        if d:
            recs.append(pd.DataFrame(d))
    df = pd.concat(recs, ignore_index=True).merge(sp, on="species_id", how="left")
    df["pft"] = df.functional_type.map(NPN2PFT)
    df = df[df.pft.notna() & (df.mean_first_yes_doy > 0)]
    df = df[
        (df.mean_numdays_since_prior_no > 0) & (df.mean_numdays_since_prior_no <= 14)
    ]
    df["year"] = df.mean_first_yes_year.astype(int)
    out = []
    for ph, ids in (("SOS", NPN_SOS), ("EOS", NPN_EOS_COL)):
        s = df[df.phenophase_id.isin(ids)]
        if (
            ph == "EOS"
        ):  # falling leaves only where no colored-leaf record for that site x species x year
            fall = df[df.phenophase_id.isin(NPN_EOS_FALL)]
            k = ["site_id", "species_id", "year"]
            fall = fall.merge(s[k].drop_duplicates(), on=k, how="left", indicator=True)
            s = pd.concat([s, fall[fall._merge == "left_only"].drop(columns="_merge")])
        # per site x species x year: earliest onset phenophase record
        s = s.sort_values("mean_first_yes_doy").drop_duplicates(
            ["site_id", "species_id", "year"]
        )
        g = s.groupby(["site_id", "pft", "year"]).agg(
            lat=("latitude", "first"),
            lon=("longitude", "first"),
            doy=("mean_first_yes_doy", "median"),
            n_species=("species_id", "nunique"),
        )
        g = g.reset_index()
        g["phase"] = ph
        out.append(g)
    o = pd.concat(out, ignore_index=True)
    o["source"] = "NPN"
    o["site"] = "npn" + o.site_id.astype(str)
    o["veg"] = o.pft
    o["date"] = pd.to_datetime(o.year.astype(str) + "-01-01") + pd.to_timedelta(
        o.doy - 1, unit="D"
    )
    return o.drop(columns="site_id")


def main():
    SMALL.mkdir(parents=True, exist_ok=True)
    ground = pd.concat([phenocam(), npn()], ignore_index=True)
    ground.to_csv(SMALL / "ground_siteyears.csv", index=False)
    pix = pd.read_parquet(
        PIX, columns=["location", "latitude", "longitude", "pft", "phase", "seasonal"]
    )
    pix = pix[pix.phase == "SOS"].drop(columns="phase")
    pix["glat"], pix["glon"] = (
        (pix.latitude * 4).round().astype(int),
        (pix.longitude * 4).round().astype(int),
    )
    sites = ground.drop_duplicates(["source", "site", "pft"])[
        ["source", "site", "veg", "pft", "lat", "lon"]
    ].copy()
    sites["glat"], sites["glon"] = (
        (sites.lat * 4).round().astype(int),
        (sites.lon * 4).round().astype(int),
    )
    m = sites.merge(
        pix.rename(columns={"pft": "pix_pft"}), on=["glat", "glon"], how="left"
    )
    m["pft_match"] = m.pft == m.pix_pft
    m["has_pixel"] = m.location.notna()
    # keep: own-PFT pixel if it exists, otherwise all other PFT datasets at that grid point
    own = m.groupby(["source", "site", "pft"]).pft_match.transform("any")
    m = m[(m.pft_match) | (~own & m.has_pixel) | (~m.has_pixel)]
    m.to_csv(SMALL / "ground_pixels.csv", index=False)
    s = m.drop_duplicates(["source", "site", "pft", "pix_pft"])
    print(
        "site-years:",
        ground.groupby(["source", "pft", "phase"]).size().unstack().to_string(),
    )
    print("sites with a pixel, by own PFT match:")
    print(
        s.groupby(["source", "pft"])
        .agg(
            n_sites=("site", "nunique"),
            with_pixel=("has_pixel", "sum"),
            own_pft=("pft_match", "sum"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
