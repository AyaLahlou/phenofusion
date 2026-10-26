#!/usr/bin/env python
"""Item 6 (PhenoCam model comparison, step A): cache what is needed to rebuild TFT/PhenoFormer inputs for new windows
without reloading the 34 GB TFT_30 BDT pickle: feature_map, scalers, cardinalities, and a validation subset of test
windows (all input arrays + the stored TFT/PhenoFormer predictions of the same rows).
Output: /burg-archive/glab/users/al4385/analysis/review/item6/phenocam_cache.pkl"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pc_cache")
G = Path("/burg-archive/glab/users/al4385")
REPO = Path("/burg-archive/home/al4385/phenofusion")
OUT = G / "analysis/review/item6/phenocam_cache.pkl"

gp = pd.read_csv(REPO / "data/review/item6/ground_pixels.csv")
pc_locs = set(
    gp[(gp.pft == "BDT") & gp.pft_match & gp.has_pixel].location.dropna().astype(int)
)
log.info(f"{len(pc_locs)} PhenoCam BDT locations")

with open(G / "data/TFT_30/merged_BDT_1982_2021.pkl", "rb") as fh:
    d = pickle.load(fh)
log.info(f"keys {list(d)}; test keys {list(d['data_sets']['test'])}")
t = d["data_sets"]["test"]
ids = np.asarray(t["id"])[:, 0].astype(str)
loc = np.array([s.split("_")[0] for s in ids]).astype(int)
rng = np.random.default_rng(0)
i_pc = np.flatnonzero(np.isin(loc, list(pc_locs)))
sel = np.concatenate([rng.permutation(i_pc)[:40], rng.permutation(len(ids))[:40]])
sel = np.unique(sel)
log.info(
    f"N test {len(ids)}; PhenoCam-location windows {len(i_pc)}; validation subset {len(sel)}"
)
sub = {k: np.asarray(v[sel]) for k, v in t.items()}
cache = dict(
    feature_map=d["feature_map"],
    scalers=d["scalers"],
    cardinalities=d["categorical_cardinalities"],
    sel=sel,
    sub=sub,
    n_test=len(ids),
)
for k, v in sub.items():
    log.info(f"  {k}: {v.shape} {v.dtype}")
del d, t
with open(G / "predictions/pretrained_1219/pred_merged_BDT_feb172025.pkl", "rb") as fh:
    pr = pickle.load(fh)
log.info(f"pred keys {list(pr)}")
cache["tft_pred"] = np.asarray(pr["predicted_quantiles"][sel])
cache["tft_n"] = len(pr["predicted_quantiles"])
del pr
with open(G / "data/phenoformer/BDT_predictions.pkl", "rb") as fh:
    pf = pickle.load(fh)
cache["pf_pred"] = np.asarray(pf["predicted_quantiles"])[sel]
cache["pf_n"] = len(pf["predicted_quantiles"])
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "wb") as fh:
    pickle.dump(cache, fh)
log.info(
    f"wrote {OUT}; tft_n {cache['tft_n']} pf_n {cache['pf_n']} n_test {cache['n_test']}"
)
