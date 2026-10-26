"""Compare the TFT_30 (benchmark) test split with the smfix TFT test split: keys, feature maps, ids (item 6)."""

import pickle
import sys
import numpy as np

for p in sys.argv[1:]:
    print("=" * 20, p)
    d = pickle.load(open(p, "rb"))
    print("top keys", list(d.keys()))
    print("feature_map", d.get("feature_map"))
    for s, v in d["data_sets"].items():
        print(
            s,
            {
                k: (getattr(a, "shape", None), getattr(a, "dtype", None))
                for k, a in v.items()
            },
        )
    t = d["data_sets"]["test"]
    ids = np.asarray(t["id"])
    print(
        "id shape",
        ids.shape,
        ids.reshape(len(ids), -1)[:3, :2],
        ids.reshape(len(ids), -1)[-2:, :2],
    )
    flat = ids.reshape(len(ids), -1)
    first = flat[:, 0].astype(str)
    loc = np.array([x.split("_")[0] for x in first])
    dt = np.array([x.split("_")[1] for x in first])
    print(
        "n loc",
        len(np.unique(loc)),
        "dates",
        dt.min(),
        dt.max(),
        "n unique dates",
        len(np.unique(dt)),
    )
    print("example dates loc0", sorted(dt[loc == loc[0]])[:12])
    print(
        "scalers",
        type(d.get("scalers")),
        list(d["scalers"].keys())[:12] if isinstance(d.get("scalers"), dict) else "",
    )
    sc = d["scalers"]
    for k in list(sc)[:3]:
        print(k, sc[k])
    for k in ("split_info",):
        if k in d:
            print(k, d[k])
    del d
