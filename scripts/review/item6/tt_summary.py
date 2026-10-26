"""Item 6: thermal-time positive control (obs_st.py, GDD base 5 degC, median GDD requirement) scored like a benchmark:
r_tt (interannual corr of predicted vs CSIF-observed SOS, per pixel), ST_tt (true response of the model to +1 degC) vs observed
ST60 (item 2, smfix). Seasonal SOS pixels. Writes data/review/item6/thermaltime_obs_st.csv."""

import pandas as pd

p = pd.read_parquet(
    "/burg-archive/glab/users/al4385/analysis/ideas/idea1/obs_st/all_pixels.parquet",
    columns=[
        "pft",
        "phase",
        "seasonal",
        "ST60",
        "ST_tt",
        "ST_tt_app",
        "r_tt",
        "sd_days",
        "n_years",
    ],
)
p = p[(p.phase == "SOS") & p.seasonal.astype(bool)]
g = p.groupby("pft").agg(
    n_pix=("ST60", "size"),
    r_tt_median=("r_tt", "median"),
    r_tt_q25=("r_tt", lambda s: s.quantile(0.25)),
    r_tt_q75=("r_tt", lambda s: s.quantile(0.75)),
    frac_r_tt_gt0_3=("r_tt", lambda s: (s > 0.3).mean()),
    ST_tt_true_median=("ST_tt", "median"),
    ST_tt_apparent_median=("ST_tt_app", "median"),
    ST60_obs_median=("ST60", "median"),
    sd_sos_days=("sd_days", "median"),
)
g.round(3).to_csv(
    "/burg-archive/home/al4385/phenofusion/data/review/item6/thermaltime_obs_st.csv"
)
print(g.round(2).to_string())
