"""Locate the native LCSIF node days inside the cubic-spline-interpolated daily record (item 6).
Within a cubic segment the 4th difference is 0; it is non-zero only where the stencil straddles a node.
Prints the day-of-month histogram of 'kink' days for a few locations."""

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

f = sys.argv[1]
tb = pq.read_table(
    f, columns=["time", "location", "sif_clear_inst"], filters=[("location", "<", 40)]
).to_pandas()
tb["time"] = pd.to_datetime(tb["time"])
allk = []
for loc, g in tb.groupby("location"):
    g = g.sort_values("time")
    x = g.sif_clear_inst.to_numpy(float)
    t = g.time.to_numpy()
    d4 = np.abs(np.diff(x, 4))
    scale = np.nanmedian(np.abs(np.diff(x))) + 1e-12
    # stencil i covers days i..i+4; a node at day i+2 gives the largest |d4| at i (centre i+2)
    k = np.flatnonzero(d4 > 1e-3 * scale) + 2
    allk.append(pd.Series(t[k]))
    if loc < 2:
        print("loc", loc, "n days", len(x), "frac kink days", len(k) / len(x))
        print(pd.Series(t[k[:60]]).dt.strftime("%Y-%m-%d").tolist())
        z = np.abs(np.diff(x, 4))
        print("d4 quantiles", np.nanquantile(z / scale, [0.1, 0.5, 0.9, 0.99]))
k = pd.concat(allk)
print("day-of-month histogram of kink days:")
print(k.dt.day.value_counts().sort_index().to_string())
# local maxima of d4 only
