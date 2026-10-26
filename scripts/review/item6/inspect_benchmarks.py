"""Describe the structure of the benchmark prediction / formatted pickles (item 6)."""

import pickle
import sys
import numpy as np
import pandas as pd


def describe(o, pre="", depth=0):
    t = type(o).__name__
    if isinstance(o, dict):
        print(f"{pre}dict[{len(o)}]")
        for i, (k, v) in enumerate(o.items()):
            if i >= 25:
                print(f"{pre}  ...")
                break
            print(f"{pre}  {k!r}:", end=" ")
            describe(v, pre + "    ", depth + 1)
    elif isinstance(o, (list, tuple)):
        print(f"{pre}{t}[{len(o)}]", end=" ")
        if len(o) and depth < 4:
            describe(o[0], pre + "    ", depth + 1)
        else:
            print()
    elif isinstance(o, np.ndarray):
        s = f"ndarray{o.shape} {o.dtype}"
        if o.size and o.dtype.kind in "fiu":
            s += f" min={np.nanmin(o):.4g} max={np.nanmax(o):.4g} mean={np.nanmean(o):.4g} nan={np.isnan(o.astype(float)).mean():.3f}"
        elif o.size:
            s += f" first={o.flat[0]!r}"
        print(s)
    elif isinstance(o, pd.DataFrame):
        print(f"DataFrame{o.shape} cols={list(o.columns)[:30]}")
        print(o.head(3).to_string()[:1500])
    elif hasattr(o, "shape"):
        print(f"{t}{tuple(o.shape)}")
    else:
        print(f"{t}: {repr(o)[:200]}")


for p in sys.argv[1:]:
    print("=" * 30, p)
    with open(p, "rb") as f:
        try:
            o = pickle.load(f)
        except Exception as e:
            print("FAILED", e)
            continue
    describe(o)
    if isinstance(o, dict):
        for k in ("id", "ids", "time_index", "dates"):
            if k in o:
                v = np.asarray(o[k])
                print(k, "examples:", v[:3], v[-3:])
