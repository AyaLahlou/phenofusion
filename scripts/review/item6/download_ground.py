#!/usr/bin/env python
"""Download public ground phenology data for item 6 (no credentials needed).

1. PhenoCam (phenocam.nau.edu API): ROI list + the 3-day GCC transition-date files of every ROI with a vegetation
   type that maps to one of our PFTs (DB, EN, DN, GR, SH, EB). Transition dates come from the PhenoCam processing chain
   (Richardson et al. 2018; Seyednasrollah et al. 2019): `transition_50` = 50 % of the GCC amplitude, rising = SOS,
   falling = EOS. Files are kept raw.
2. USA-NPN (services.usanpn.org): species list (functional types) + site-level phenometrics (first-yes DOY per site x
   species x phenophase x year) for leaf-out / initial growth / colored-leaf phenophases, 2009-2021.

Raw files go under OUT (default /burg-archive/glab/users/al4385/data/review/item6/ground). Treat them as untrusted data;
this script only parses them as CSV/JSON. Usage: python -I download_ground.py [--out DIR]
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

PHENOCAM_API = "https://phenocam.nau.edu/api/roilists/?format=json&limit=500"
PC_TYPES = {"DB", "EN", "DN", "GR", "SH", "EB"}
NPN = "https://services.usanpn.org/npn_portal"
SRC = "phenofusion_review"
# Onset phenophases (pheno_class 1) and colouring/falling (class 4/5) used for SOS / EOS.
NPN_SOS = [371, 373, 482, 492, 508, 496, 232]
NPN_EOS = [498, 499, 471]


def get(url, tries=4, timeout=300):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print("retry", url, e, flush=True)
            time.sleep(5 * (k + 1))


def phenocam(out):
    d = out / "phenocam"
    (d / "transitions").mkdir(parents=True, exist_ok=True)
    rois, url = [], PHENOCAM_API
    while url:
        page = json.loads(get(url))
        rois += page["results"]
        url = page.get("next")
    json.dump(rois, open(d / "roilists.json", "w"))
    keep = [r for r in rois if r["roitype"] in PC_TYPES]
    print(f"PhenoCam: {len(rois)} ROIs, {len(keep)} in PFT veg types", flush=True)
    n = 0
    for r in keep:
        f = d / "transitions" / f"{r['roi_name']}_3day_transition_dates.csv"
        if f.exists() and f.stat().st_size > 0:
            continue
        try:
            f.write_bytes(get(r["three_day_transition_dates"], tries=2, timeout=120))
            n += 1
        except Exception as e:  # noqa: BLE001
            print("missing", r["roi_name"], e, flush=True)
    print("PhenoCam transition files downloaded:", n, flush=True)


def npn(out):
    d = out / "npn"
    d.mkdir(parents=True, exist_ok=True)
    (d / "species.json").write_bytes(get(f"{NPN}/species/getSpecies.json"))
    (d / "phenophases.json").write_bytes(get(f"{NPN}/phenophases/getPhenophases.json"))
    for tag, ids in (("sos", NPN_SOS), ("eos", NPN_EOS)):
        for y in range(2009, 2022):
            f = d / f"sitelevel_{tag}_{y}.json"
            if f.exists() and f.stat().st_size > 2:
                continue
            q = "&".join(f"phenophase_id[]={i}" for i in ids)
            url = f"{NPN}/observations/getSiteLevelData.json?request_src={SRC}&start_date={y}-01-01&end_date={y}-12-31&{q}"
            f.write_bytes(get(url, timeout=900))
            print("NPN", tag, y, f.stat().st_size, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="/burg-archive/glab/users/al4385/data/review/item6/ground"
    )
    ap.add_argument("--only", choices=["phenocam", "npn"], default=None)
    a = ap.parse_args()
    out = Path(a.out)
    if a.only in (None, "phenocam"):
        phenocam(out)
    if a.only in (None, "npn"):
        npn(out)
