import pandas as pd
import numpy as np
from itertools import product
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def get_analysis_df(data, preds, coord_path):
    """Match original data to predictions and corresponding coordinates.

    Parameters
    ----------
    data_path : str
        path to original processed data dictionnary
    pred_path : str
        path to predictions dictionnary
    coord_path : str
        path to dataframe matching location index to coordinates

    Returns
    -------
    pd.Dataframe
        dataframe with groundtruth values, predictions and corresponding time and coordinates

    """
    coords = pd.read_parquet(coord_path)
    coords = coords.drop_duplicates()
    # get location ID and groundtruth CSIF from original data
    df = pd.DataFrame(
        {
            "Index": data["data_sets"]["test"]["id"].flatten(),
            "Flattened_Values": data["data_sets"]["test"]["target"].flatten(),
        }
    )
    # retrive future drivers map from predictions
    df["pred_05"] = preds["predicted_quantiles"][:, :, 1].flatten()
    df[["location_id", "time_id"]] = df["Index"].str.split("_", n=1, expand=True)
    df["location_id"] = df["location_id"].astype(int)
    df["time_id"] = pd.to_datetime(df["time_id"])
    df = df.sort_values(by=["location_id", "time_id"])
    df["doy"] = df["time_id"].dt.dayofyear
    df["year"] = df["time_id"].dt.year
    df["month"] = df["time_id"].dt.month
    df["day"] = df["time_id"].dt.day
    df = df.rename(
        columns={
            "Flattened_Values": "CSIF",
            "location_id": "location",
            "time_id": "time",
        }
    )
    df = df.drop(columns=["Index"])
    df = pd.merge(coords, df, on="location", how="left")
    return df


def spring_series_indices(df, month_start, month_end, specific_year=None):
    """
    output: indexes of time series predicted in the spring time.
    """
    res_indices = []
    year_range = range(2002, 2011)
    if specific_year is not None:
        year_range = [specific_year]

    for year in year_range:
        dt_start = datetime(year, month_start, 10)
        dt_end = datetime(year, month_end, 30)
        df_current_year = df[df["year"] == year]
        filtered_df = df_current_year.groupby("location").filter(
            lambda x: (x["time"].iloc[0] >= dt_start) and (x["time"].iloc[0] <= dt_end)
        )
        indices_array = np.unique(filtered_df[::30].index.to_numpy())
        print("indeces_array", indices_array)
        res_indices.extend(np.round(indices_array / 30).astype(int))

    return res_indices

    # TO DO : case for EOS


def max_attention_window(preds, index):
    # get array of mean attention of all horizons at each timesteps
    att_array = np.mean(preds["attention_scores"][index], axis=0)

    # Initialize variables
    max_sum = -np.inf
    best_start_index = None

    # Slide over the columns
    for i in range(396 - 30):  # We slide up to 365th index for a 30-day window
        current_sum = np.sum(att_array[i : i + 30])
        if current_sum > max_sum:
            max_sum = current_sum
            best_start_index = i

    return best_start_index


def plot_attention(
    data_path, pred_path, coord_path, month_start, month_end, lat_min, lat_max
):
    with open(data_path, "rb") as fp:
        data = pickle.load(fp)
    with open(pred_path, "rb") as fp_2:
        preds = pickle.load(fp_2)

    df = get_analysis_df(data, preds, coord_path)
    fp.close()

    df = df[df["latitude"] < lat_max]
    df = df[df["latitude"] > lat_min]

    select_indices = spring_series_indices(df, month_start, month_end)

    att_mat_list = []
    csif_ts = []

    for index in select_indices:
        att_mat_list.append(preds["attention_scores"][index])
        csif_ts.append(
            np.concatenate(
                [
                    np.array(
                        data["data_sets"]["test"]["historical_ts_numeric"][index][:, 0]
                    ),
                    np.array(data["data_sets"]["test"]["target"][index]),
                ],
                axis=0,
            )
        )

    stacked_att = np.stack(att_mat_list, axis=0)
    att_mean_array = np.mean(stacked_att, axis=0)

    stacked_csif_ts = np.stack(csif_ts, axis=0)
    csif_mean_array = np.mean(stacked_csif_ts, axis=0)

    plt.plot(csif_mean_array)
    fp_2.close()

    plt.figure(figsize=(20, 3))
    cmap0 = LinearSegmentedColormap.from_list("", ["white", "red"])
    sns.heatmap(
        att_mean_array, cmap=cmap0
    )  # You can choose different colormaps like 'coolwarm', 'Blues', etc.
    plt.axvline(x=365, color="black", linestyle="--", linewidth=1)
    # Adding titles and labels as needed
    plt.xlabel("Time")
    plt.ylabel("Horizon")
    plt.title(data_path.split("/")[-1].split(".")[0] + season)


def impute_nearby(df, lat_range=0.5, lon_range=0.5):
    for i, row in df.iterrows():
        if (
            not row[
                [
                    "hist_tmin",
                    "hist_tmax",
                    "hist_rad",
                    "hist_precip",
                    "hist_photo",
                    "hist_sm",
                ]
            ]
            .isnull()
            .all()
        ):
            # Find rows that are close in latitude and longitude and have NaN values
            mask = (
                (df["latitude"] >= row["latitude"] - lat_range)
                & (df["latitude"] <= row["latitude"] + lat_range)
                & (df["longitude"] >= row["longitude"] - lon_range)
                & (df["longitude"] <= row["longitude"] + lon_range)
                & df[
                    [
                        "hist_tmin",
                        "hist_tmax",
                        "hist_rad",
                        "hist_precip",
                        "hist_photo",
                        "hist_sm",
                    ]
                ]
                .isnull()
                .all(axis=1)
            )
            # Impute the NaN values with the current row's values
            df.loc[
                mask,
                [
                    "hist_tmin",
                    "hist_tmax",
                    "hist_rad",
                    "hist_precip",
                    "hist_photo",
                    "hist_sm",
                ],
            ] = row[
                [
                    "hist_tmin",
                    "hist_tmax",
                    "hist_rad",
                    "hist_precip",
                    "hist_photo",
                    "hist_sm",
                ]
            ].values

    return df


def concatenate(df_list):
    # Group by latitude, longitude (can have points for many years).
    df = pd.concat(df_list, ignore_index=True)
    grouped_df = df.groupby(["latitude", "longitude"]).mean().reset_index()

    hist_temp = grouped_df["hist_tmin"] + grouped_df["hist_tmax"]
    hist_sol = grouped_df["hist_rad"] + grouped_df["hist_photo"]
    hist_p = grouped_df["hist_precip"] + grouped_df["hist_sm"]

    grouped_df["hist_temp"] = hist_temp
    grouped_df["hist_sol"] = hist_sol
    grouped_df["hist_p"] = hist_p

    # Columns to normalize
    columns_to_normalize = ["hist_temp", "hist_sol", "hist_p"]

    # Normalize the selected columns
    df_normalized = grouped_df[columns_to_normalize].div(
        grouped_df[columns_to_normalize].sum(axis=1), axis=0
    )

    grouped_df["hist_temp"] = df_normalized["hist_temp"]
    grouped_df["hist_sol"] = df_normalized["hist_sol"]
    grouped_df["hist_p"] = df_normalized["hist_p"]

    desired_lat_values = np.linspace(90.0, -89.75, 720)
    desired_lon_values = np.linspace(-180.0, 179.75, 1440)
    # Generate all combinations of values
    combinations = list(product(desired_lat_values, desired_lon_values))
    # Create a DataFrame from the combinations
    df_fullcoord = pd.DataFrame(combinations, columns=["latitude", "longitude"])
    full = pd.merge(df_fullcoord, grouped_df, on=["latitude", "longitude"], how="left")

    return full


def colorize(full):
    a_data = np.reshape(full["hist_temp"], (720, 1440))
    b_data = np.reshape(full["hist_sol"], (720, 1440))
    c_data = np.reshape(full["hist_p"], (720, 1440))

    w = 255
    scale = 100
    x_color = a_data * w / float(scale)  # temp
    y_color = b_data * w / float(scale)  # sol
    z_color = c_data * w / float(scale)

    r_arr = []  # temp - Y
    for i in x_color:  # temp
        lon = []
        for j in i:
            r = math.fabs(w - j) / w
            if np.isnan(j):
                lon.append(255)
            else:
                lon.append(j)
        r_arr.append(lon)
    g_arr = []  # rad - R
    for i in y_color:  # sol
        lon = []
        for j in i:
            g = math.fabs(w - j) / w
            if np.isnan(j):
                lon.append(255)
            else:
                lon.append(j)
        g_arr.append(lon)

    b_arr = []  # precip - B
    for i in z_color:
        lon = []
        for j in i:
            b = math.fabs(w - j) / w
            if np.isnan(j):
                lon.append(255)
            else:
                lon.append(j)
        b_arr.append(lon)

    return [b_arr, g_arr, r_arr]  # [#sol,#temp,#precp] --> [cyan, magenta, yellow]


def plot_map_robinson(rgb_list, title=None):
    # Ensure the rgb_list contains exactly 3 elements (R, G, B channels)
    if len(rgb_list) != 3:
        raise ValueError(
            "rgb_list must contain exactly 3 elements corresponding to R, G, B channels."
        )

    # Stack the RGB channels along the last axis to form an RGB image
    rgb_data = np.stack(rgb_list, axis=-1)

    # Verify the shape of the RGB data (should be 2D grid with 3 color channels)
    if rgb_data.ndim != 3 or rgb_data.shape[-1] != 3:
        raise ValueError("Stacked RGB data should have a shape of (height, width, 3).")

    # Define latitude and longitude values for ticks
    latitudes = np.linspace(90.0, -90.0, rgb_data.shape[0])  # Match the grid shape
    longitudes = np.linspace(-180.0, 180.0, rgb_data.shape[1])  # Match the grid shape

    # Create a new figure and plot
    plt.figure(figsize=(16, 8))

    # Create a Cartopy projection using Robinson projection
    ax = plt.axes(projection=ccrs.Robinson())

    # Plot coastlines for reference
    ax.coastlines()

    # Add gridlines
    ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)

    # Plot the RGB data using the `transform` argument for the PlateCarree projection
    plt.imshow(
        rgb_data,
        extent=[-180, 180, -90, 90],
        origin="upper",
        transform=ccrs.PlateCarree(),
        interpolation="none",
    )
    if title is not None:
        plt.title(title)
    # Display the plot

    plt.savefig("/burg/home/al4385/figures/" + title + ".png")
    plt.show()


def impute_nearby_leg(df, lat_range=0.5, lon_range=0.5):
    for i, row in df.iterrows():
        if not row[["attention"]].isnull().all():
            # Find rows that are close in latitude and longitude and have NaN values
            mask = (
                (df["latitude"] >= row["latitude"] - lat_range)
                & (df["latitude"] <= row["latitude"] + lat_range)
                & (df["longitude"] >= row["longitude"] - lon_range)
                & (df["longitude"] <= row["longitude"] + lon_range)
                & df[["attention"]].isnull().all(axis=1)
            )
            # Impute the NaN values with the current row's values
            df.loc[mask, ["attention"]] = row[["attention"]].values

    return df


def legacy_df(data, preds, coord_path, year, season, size=0.25):
    df = get_analysis_df(data, preds, coord_path)

    eos_dic = {
        (60, 90): (7, 8),
        (35, 60): (8, 9),
        (30, 35): (8, 9),
        (20, 30): (9, 9),
        (15, 20): (9, 10),
        (5, 15): (9, 10),
        (0, 5): (7, 9),
        (-3, 0): (5, 6),
        (-5, -3): (4, 5),
        (-10, -5): (4, 5),
        (-24, -10): (3, 4),
        (-30, -24): (4, 5),
        (-40, -30): (7, 8),
        (-50, -40): (4, 5),
        (-61, -50): (5, 6),
    }
    sos_dic = {
        (60, 90): (5, 6),
        (55, 60): (4, 5),
        (40, 55): (3, 5),
        (30, 40): (4, 6),
        (20, 30): (5, 6),
        (15, 20): (6, 7),
        (12, 15): (5, 6),
        (10, 12): (4, 5),
        (7, 10): (3, 4),
        (3, 7): (2, 4),
        (0, 3): (3, 4),
        (-2, 0): (5, 6),
        (-5, -2): (6, 7),
        (-13, -5): (8, 9),
        (-20, -13): (9, 11),
        (-26, -20): (9, 10),
        (-30, -26): (8, 9),
        (-40, -30): (7, 9),
        (-50, -40): (7, 8),
        (-61, -50): (8, 9),
    }

    select_indices = []
    if season == "eos":
        for lats, months in eos_dic.items():
            df_current = df[df["latitude"] > lats[0]]
            df_current = df_current[df_current["latitude"] <= lats[1]]
            select_indices = np.concatenate(
                (
                    select_indices,
                    spring_series_indices(df_current, months[0], months[1], year),
                )
            ).astype(int)
    elif season == "sos":
        for lats, months in sos_dic.items():
            df_current = df[df["latitude"] > lats[0]]
            df_current = df_current[df_current["latitude"] <= lats[1]]
            select_indices = np.concatenate(
                (
                    select_indices,
                    spring_series_indices(df_current, months[0], months[1], year),
                )
            ).astype(int)

    df_attention_map = pd.DataFrame(columns=["location", "attention"])

    for index in select_indices:
        window_start = max_attention_window(preds, index)

        location = np.int64(data["data_sets"]["test"]["id"][index][0].split("_")[0])

        new_row = pd.DataFrame({"location": [location], "attention": [window_start]})
        df_attention_map = pd.concat([df_attention_map, new_row], ignore_index=True)

    coords = pd.read_parquet(coord_path)
    coords = coords.drop_duplicates()
    df_coord_att = pd.merge(coords, df_attention_map, on="location", how="left")
    imputed_coord_att = impute_nearby_leg(df_coord_att, size, size)
    return imputed_coord_att


def concatenate_legacy(df_list):
    # Group by latitude, longitude (can have points for many years).
    df = pd.concat(df_list, ignore_index=True)
    grouped_df = df.groupby(["latitude", "longitude"]).median().reset_index()

    desired_lat_values = np.linspace(90.0, -89.75, 720)
    desired_lon_values = np.linspace(-180.0, 179.75, 1440)
    # Generate all combinations of values
    combinations = list(product(desired_lat_values, desired_lon_values))
    # Create a DataFrame from the combinations
    df_fullcoord = pd.DataFrame(combinations, columns=["latitude", "longitude"])
    full = pd.merge(df_fullcoord, grouped_df, on=["latitude", "longitude"], how="left")

    return full


def plot_map_legacy(channel_data, season, pft):
    # Convert channel_data to a numeric array if it's not already
    channel_data = np.asarray(channel_data, dtype=np.float32)

    # Ensure the input is a 2D array (single channel)
    if channel_data.ndim != 2:
        raise ValueError("Input data must be a 2D array for a single channel.")

    # Define latitude and longitude values for ticks
    latitudes = np.linspace(90.0, -90.0, channel_data.shape[0])  # Match the grid shape
    longitudes = np.linspace(
        -180.0, 180.0, channel_data.shape[1]
    )  # Match the grid shape

    # Create a new figure and plot
    plt.figure(figsize=(16, 8))

    # Create a Cartopy projection using Robinson projection
    ax = plt.axes(projection=ccrs.Robinson())

    # Plot coastlines for reference
    ax.coastlines()

    # Add gridlines
    ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)

    # Plot the single channel data using the `transform` argument for the PlateCarree projection
    img = plt.imshow(
        channel_data,
        extent=[-180, 180, -90, 90],
        origin="upper",
        cmap="nipy_spectral",
        transform=ccrs.PlateCarree(),
        interpolation="none",
    )

    # Add a colorbar
    plt.colorbar(img, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    plt.title("Memory Effects During" + str(season))
    plt.savefig("/burg/home/al4385/figures/Memory_" + season + "_" + pft + ".png")
    # Display the plot
    plt.show()


def att_drivers_df(data, preds, coord_path, year, season, size=0.25):
    df = get_analysis_df(data, preds, coord_path)

    eos_dic = {
        (60, 90): (7, 8),
        (35, 60): (8, 9),
        (30, 35): (8, 9),
        (20, 30): (9, 9),
        (15, 20): (9, 10),
        (5, 15): (9, 10),
        (0, 5): (7, 9),
        (-3, 0): (5, 6),
        (-5, -3): (4, 5),
        (-10, -5): (4, 5),
        (-24, -10): (3, 4),
        (-30, -24): (4, 5),
        (-40, -30): (7, 8),
        (-50, -40): (4, 5),
        (-61, -50): (5, 6),
    }
    sos_dic = {
        (60, 90): (5, 6),
        (55, 60): (4, 5),
        (40, 55): (3, 5),
        (30, 40): (4, 6),
        (20, 30): (5, 6),
        (15, 20): (6, 7),
        (12, 15): (5, 6),
        (10, 12): (4, 5),
        (7, 10): (3, 4),
        (3, 7): (2, 4),
        (0, 3): (3, 4),
        (-2, 0): (5, 6),
        (-5, -2): (6, 7),
        (-13, -5): (8, 9),
        (-20, -13): (9, 11),
        (-26, -20): (9, 10),
        (-30, -26): (8, 9),
        (-40, -30): (7, 9),
        (-50, -40): (7, 8),
        (-61, -50): (8, 9),
    }

    select_indices = []
    if season == "eos":
        for lats, months in eos_dic.items():
            df_current = df[df["latitude"] > lats[0]]
            df_current = df_current[df_current["latitude"] <= lats[1]]
            select_indices = np.concatenate(
                (
                    select_indices,
                    spring_series_indices(df_current, months[0], months[1], year),
                )
            ).astype(int)
    elif season == "sos":
        for lats, months in sos_dic.items():
            df_current = df[df["latitude"] > lats[0]]
            df_current = df_current[df_current["latitude"] <= lats[1]]
            select_indices = np.concatenate(
                (
                    select_indices,
                    spring_series_indices(df_current, months[0], months[1], year),
                )
            ).astype(int)

    df_attention_map = pd.DataFrame(
        columns=[
            "location",
            "hist_tmin",
            "hist_tmax",
            "hist_rad",
            "hist_precip",
            "hist_photo",
            "hist_sm",
        ]
    )

    for index in select_indices:
        window_start = max_attention_window(preds, index)
        if window_start <= 335:
            tmin_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + 30, 1
                ]
            )
            tmax_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + 30, 2
                ]
            )
            rad_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + 30, 3
                ]
            )
            precip_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + 30, 4
                ]
            )
            photo_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + 30, 5
                ]
            )
            sm_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + 30, 6
                ]
            )
            location = np.int64(data["data_sets"]["test"]["id"][index][0].split("_")[0])

            new_row = pd.DataFrame(
                {
                    "location": [location],
                    "hist_tmin": [tmin_weight],
                    "hist_tmax": [tmax_weight],
                    "hist_rad": [rad_weight],
                    "hist_precip": [precip_weight],
                    "hist_photo": [photo_weight],
                    "hist_sm": [sm_weight],
                }
            )
            df_attention_map = pd.concat([df_attention_map, new_row], ignore_index=True)

    coords = pd.read_parquet(coord_path)
    coords = coords.drop_duplicates()
    df_coord_att = pd.merge(coords, df_attention_map, on="location", how="left")
    imputed_coord_att = impute_nearby(df_coord_att, size, size)
    return imputed_coord_att


season = "eos"  # or 'eos'
year = 1985
data_directory = "/burg/glab/users/al4385/data/TFT_30_40years/"
pred_dir = "/burg/glab/users/al4385/predictions/TFT_30_40years/"
coord_dir = "/burg/glab/users/al4385/data/coordinates/"

cluster_names = [
    "BDT_50_20",
    "BDT_-20_-60",
    "BDT_-20_20",
    "BDT_50_90",
    "BET",
    "NET",
    "NDT",
]

df_list_2002 = []
for cluster in cluster_names:
    data_path = data_directory + cluster + ".pickle"
    pred_path = pred_dir + "pred_" + cluster + ".pkl"  # validation outputs
    coord_path = coord_dir + cluster + ".parquet"

    with open(data_path, "rb") as fp:
        data = pickle.load(fp)
    with open(pred_path, "rb") as fp_2:
        preds = pickle.load(fp_2)

    df_coord_att = att_drivers_df(data, preds, coord_path, year, season, 0.5)
    df_list_2002.append(df_coord_att)
    fp.close()
    fp_2.close()

full_2002 = concatenate(df_list_2002)

season = "eos"  # or 'eos'
year = 2020
data_directory = "/burg/glab/users/al4385/data/TFT_30_40years/"
pred_dir = "/burg/glab/users/al4385/predictions/TFT_30_40years/"
coord_dir = "/burg/glab/users/al4385/data/coordinates/"

cluster_names = [
    "BDT_50_20",
    "BDT_-20_-60",
    "BDT_-20_20",
    "BDT_50_90",
    "BET",
    "NET",
    "NDT",
]

df_list_2010 = []
for cluster in cluster_names:
    data_path = data_directory + cluster + ".pickle"
    pred_path = pred_dir + "pred_" + cluster + ".pkl"  # validation outputs
    coord_path = coord_dir + cluster + ".parquet"

    with open(data_path, "rb") as fp:
        data = pickle.load(fp)
    with open(pred_path, "rb") as fp_2:
        preds = pickle.load(fp_2)

    df_coord_att = att_drivers_df(data, preds, coord_path, year, season, 0.5)
    df_list_2010.append(df_coord_att)
    fp.close()
    fp_2.close()

full_2010 = concatenate(df_list_2010)

difference = full_2002 - full_2010

temp_difference = np.reshape(difference["hist_temp"], (720, 1440))
solar_radiation_difference = np.reshape(difference["hist_sol"], (720, 1440))
water_availability_difference = np.reshape(difference["hist_p"], (720, 1440))


from matplotlib.colors import TwoSlopeNorm


def plot_map_difference(channel_data, season, title):
    # Convert channel_data to a numeric array if it's not already
    channel_data = np.asarray(channel_data, dtype=np.float32)
    # Ensure the input is a 2D array (single channel)
    if channel_data.ndim != 2:
        raise ValueError("Input data must be a 2D array for a single channel.")
    # Define latitude and longitude values for ticks
    latitudes = np.linspace(90.0, -90.0, channel_data.shape[0])  # Match the grid shape
    longitudes = np.linspace(
        -180.0, 180.0, channel_data.shape[1]
    )  # Match the grid shape
    # Create a new figure and plot
    plt.figure(figsize=(16, 8))
    # Create a Cartopy projection using Robinson projection
    ax = plt.axes(projection=ccrs.Robinson())
    # Plot coastlines for reference
    ax.coastlines()
    # Add gridlines
    ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    # Plot the single channel data using the `transform` argument for the PlateCarree projection
    norm = TwoSlopeNorm(vcenter=0)
    img = plt.imshow(
        channel_data,
        extent=[-180, 180, -90, 90],
        origin="upper",
        norm=norm,
        cmap="RdBu_r",
        transform=ccrs.PlateCarree(),
        interpolation="none",
    )
    # Add a colorbar
    plt.colorbar(img, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    plt.title(title + "sensitivity difference during" + str(season) + "\n")
    plt.savefig(
        "/burg/home/al4385/figures/"
        + title
        + "_sensitivity_difference_1982_2022_"
        + season
        + "10_28_2024.png"
    )
    # Display the plot
    plt.show()


plot_map_difference(temp_difference, season, "temp")
plot_map_difference(solar_radiation_difference, season, "solar")
plot_map_difference(water_availability_difference, season, "water")
