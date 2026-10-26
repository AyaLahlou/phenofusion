"""Print |d2|,|d3|,|d4| (relative) per day for one location, Jan-Apr 1990, to identify the interpolation nodes."""

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

tb = (
    pq.read_table(
        sys.argv[1],
        columns=["time", "location", "sif_clear_inst"],
        filters=[("location", "==", 3)],
    )
    .to_pandas()
    .sort_values("time")
)
x = tb.sif_clear_inst.to_numpy(float)
t = pd.to_datetime(tb.time).dt.strftime("%m-%d").to_numpy()
s = np.nanmedian(np.abs(np.diff(x)))
d2 = np.r_[np.nan, np.diff(x, 2), np.nan] / s
d3 = np.r_[np.nan, np.diff(x, 3), np.nan, np.nan] / s
i0 = np.flatnonzero(pd.to_datetime(tb.time).to_numpy() >= np.datetime64("1990-01-01"))[
    0
]
for i in range(i0, i0 + 120):
    print(t[i], f"{x[i]:.5f}", f"d2={d2[i]:+.4f}", f"d3={d3[i]:+.4f}")
