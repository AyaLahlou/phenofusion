import pandas as pd
import numpy as np
from itertools import product
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse


from scipy.stats import linregress


def normalize_csif(group):
    csif_min = group["CSIF"].min()
    csif_max = group["CSIF"].max()
    group["CSIF_normalized"] = (group["CSIF"] - csif_min) / (csif_max - csif_min)
    return group


def compute_slope(group):
    # Create a time index from 0 to len(group)-1
    x = range(len(group))
    y = group["CSIF"].values
    slope, _, _, _, _ = linregress(x, y)
    return pd.Series({"slope": slope})


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

    # get location ID and grountruth CSIF from original data
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
    year_range = range(1982, 2022)
    if specific_year is not None:
        year_range = [specific_year]

    for year in year_range:
        dt_start = datetime(year, month_start, 1)
        dt_end = datetime(year, month_end, 30)

        ###### THIS MAY BE THE ISSUE ########
        # filtered_df =df_current_year.groupby('location').filter(lambda x: (x['time'].iloc[0] >= dt_start ) and (x['time'].iloc[0] <= dt_end))
        #####################################
        date_mask = (df["time"] >= dt_start) & (df["time"] <= dt_end)
        filtered_df = df[date_mask]

        indices_array = np.unique(filtered_df[::30].index.to_numpy())
        print("indices_array", indices_array)
        res_indices.extend(np.round(indices_array / 30).astype(int))

    return res_indices


# TO DO : case for EOS


def max_attention_window(preds, index, forecast_window=30):
    # get array of mean attention of all horizons at each timesteps

    att_array = np.mean(preds["attention_scores"][index], axis=0)

    # Initialize variables
    max_sum = -np.inf
    best_start_index = None

    # Slide over the columns
    for i in range(
        396 - forecast_window
    ):  # We slide up to 365th index for a 30-day window
        current_sum = np.sum(att_array[i : i + forecast_window])
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
    plt.title(data_path.split("/")[-1].split(".")[0])


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
    z_color = c_data * w / float(scale)  # precip

    r_arr = []  # temp - Y
    for i in x_color:  # temp
        lon = []
        for j in i:
            if np.isnan(j):
                lon.append(255)
            else:
                lon.append(j)
        r_arr.append(lon)
    g_arr = []  # sol
    for i in y_color:  # sol
        lon = []
        for j in i:
            if np.isnan(j):
                lon.append(255)
            else:
                lon.append(j)
        g_arr.append(lon)

    b_arr = []  # precip
    for i in z_color:
        lon = []
        for j in i:
            if np.isnan(j):
                lon.append(255)
            else:
                lon.append(j)
        b_arr.append(lon)

    return [b_arr, g_arr, r_arr]  # [#precip,#sol,#temp] --> [cyan, magenta, yellow]


def plot_map_robinson(rgb_list, output_path, title=None):
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

    # Create a new figure and plot
    plt.figure(figsize=(16, 8))

    # Create a Cartopy projection using Robinson projection
    ax = plt.axes(projection=ccrs.Robinson())

    # Plot coastlines for reference
    ax.coastlines()

    # Add gridlines
    ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    if title is not None:
        plt.title(title)
    # Plot the RGB data using the `transform` argument for the PlateCarree projection
    plt.imshow(
        rgb_data,
        extent=[-180, 180, -90, 90],
        origin="upper",
        transform=ccrs.PlateCarree(),
        interpolation="none",
    )

    # Display the plot

    plt.savefig(output_path + title + ".png")
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


def legacy_df(
    index_list, data, preds, coord_path, output_path, size=0.25, forecast_window=30
):
    df_attention_map = pd.DataFrame(columns=["location", "attention"])

    for index in index_list:
        window_start = max_attention_window(
            preds, index, forecast_window=forecast_window
        )

        location = np.int64(data["data_sets"]["test"]["id"][index][0].split("_")[0])

        new_row = pd.DataFrame({"location": [location], "attention": [window_start]})
        df_attention_map = pd.concat([df_attention_map, new_row], ignore_index=True)

    coords = pd.read_parquet(coord_path)
    coords = coords.drop_duplicates()
    df_coord_att = pd.merge(coords, df_attention_map, on="location", how="left")
    imputed_coord_att = impute_nearby_leg(df_coord_att, size, size)
    imputed_coord_att.to_csv(output_path)


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

    # eos_dic={(60,90):(7,8), (35,60):(8,9),(30,35):(8,9), (20,30):(9,9), (15,20):(9,10),(5,15):(9,10), (0,5):(7,9), (-3, 0):(5,6),(-5, -3):(4,5),(-10, -5):(4,5), (-24, -10):(3,4), (-30, -24):(4,5), (-40, -30):(7,8),(-50, -40):(4,5), (-61, -50):(5,6)}
    # sos_dic={(60,90):(5,6),(55,60):(4,5),(40,55):(3,5),(30,40):(4,6),(20,30):(5,6), (15,20):(6,7),(12,15):(5,6), (10,12):(4,5), (7,10):(3,4),(3,7):(2,4),(0,3):(3,4), (-2, 0):(5,6),(-5, -2):(6,7), (-13, -5):(8,9), (-20, -13):(9,11), (-26, -20):(9,10),(-30, -26):(8,9), (-40, -30):(7,9), (-50, -40):(7,8),(-61, -50):(8,9)}
    eos_dic = {
        (60, 90): (8, 9),
        (35, 60): (9, 10),
        (30, 35): (9, 10),
        (20, 30): (10, 11),
        (15, 20): (10, 11),
        (5, 15): (10, 11),
        (0, 5): (9, 10),
        (-3, 0): (6, 7),
        (-5, -3): (5, 6),
        (-10, -5): (5, 6),
        (-24, -10): (4, 5),
        (-30, -24): (5, 6),
        (-40, -30): (8, 9),
        (-50, -40): (5, 6),
        (-61, -50): (6, 7),
    }

    sos_dic = {
        (60, 90): (6, 7),
        (55, 60): (5, 6),
        (40, 55): (4, 6),
        (30, 40): (5, 7),
        (20, 30): (6, 7),
        (15, 20): (7, 8),
        (12, 15): (6, 7),
        (10, 12): (5, 6),
        (7, 10): (4, 5),
        (3, 7): (3, 5),
        (0, 3): (4, 5),
        (-2, 0): (6, 7),
        (-5, -2): (7, 8),
        (-13, -5): (9, 10),
        (-20, -13): (10, 12),
        (-26, -20): (10, 11),
        (-30, -26): (9, 10),
        (-40, -30): (8, 10),
        (-50, -40): (8, 9),
        (-61, -50): (9, 10),
    }
    # TODO: write function
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
    else:
        raise ValueError("Season must be 'sos' or 'eos'")

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
                    index, window_start : window_start + 30, 3
                ]
            )  # 5 is photoperiod, 3 is rad
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


def get_attention_weights_df(index_list, preds, data, forecast_window=30):
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
    for index in index_list:
        window_start = max_attention_window(preds, index, forecast_window)
        if window_start <= 365 - forecast_window:
            tmin_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + forecast_window, 1
                ]
            )
            tmax_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + forecast_window, 2
                ]
            )
            rad_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + forecast_window, 3
                ]
            )
            precip_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + forecast_window, 4
                ]
            )
            photo_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + forecast_window, 5
                ]
            )  # 5 is photoperiod, 3 is rad
            sm_weight = np.median(
                preds["historical_selection_weights"][
                    index, window_start : window_start + forecast_window, 6
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

    return df_attention_map


def save_imput_coord_att(
    index_list, data, preds, coord_path, output_path, forecast_window=30
):
    df_attention_map = get_attention_weights_df(
        index_list, preds, data, forecast_window=forecast_window
    )
    coords = pd.read_parquet(coord_path)
    coords = coords.drop_duplicates()
    df_coord_att = pd.merge(coords, df_attention_map, on="location", how="left")
    imputed_coord_att = impute_nearby(df_coord_att, 0.5, 0.5)
    imputed_coord_att.to_csv(output_path)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate driver attention maps using per-pixel phenology."
    )
    parser.add_argument("--PFT", type=str, required=False, help="PFT")
    parser.add_argument(
        "--pred_path", type=str, required=True, help="Path to predictions pickle file"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data pickle file"
    )
    parser.add_argument(
        "--coord_path", type=str, required=True, help="Path to coordinates parquet file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save output maps"
    )

    parser.add_argument(
        "--forecast_window_length", type=int, required=True, help="Forecast window size"
    )

    args = parser.parse_args()

    min_diff = 0.20
    min_slope = 0.002
    if args.PFT:
        PFT = args.PFT
        if PFT == "BET" or PFT == "SHR":
            min_diff = 0.05
            min_slope = 0.001
    # 1. Load all necessary data
    print("Loading data, predictions, and coordinates...")
    with open(args.data_path, "rb") as fp:
        data = pickle.load(fp)
    with open(args.pred_path, "rb") as fp_2:
        preds = pickle.load(fp_2)

    coord_path = args.coord_path
    output_path = args.output_path
    # 2. Prepare main analysis dataframe
    print("Preparing main analysis dataframe...")
    df = get_analysis_df(data, preds, coord_path)
    # get indices of SOS samples and EOS samples

    batch_size = 30
    SOS_index = []
    EOS_index = []
    # Iterate over DataFrame in batches of 30 rows

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]

        # Special case for BET PFT: use latitude and DOY-based detection
        if args.PFT and args.PFT == "BET":
            lat = batch_df["latitude"].iloc[0]
            doy = batch_df["doy"].iloc[0]

            if lat >= 27 and lat <= 30.5:
                if doy >= 50 and doy <= 150:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 200 and doy <= 300:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 25.75 and lat <= 27:
                if doy >= 25 and doy <= 150:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 210 and doy <= 310:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 20 and lat <= 25.5:
                if doy >= 40 and doy <= 170:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 220 and doy <= 315:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 13 and lat <= 19.75:
                if doy >= 25 and doy <= 160:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 230 and doy <= 315:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 10 and lat <= 12.75:
                if doy >= 25 and doy <= 160:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 240 and doy <= 325:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 9 and lat <= 9.75:
                if doy >= 10 and doy <= 110:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 250 and doy <= 315:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 7 and lat <= 9:
                if doy >= 10 and doy <= 90:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 260 and doy <= 325:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 4 and lat <= 7:
                if doy >= 25 and doy <= 75:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 260 and doy <= 325:
                    EOS_index.append(batch_df.index[0])
            elif lat >= 2 and lat <= 4:
                if doy >= 20 and doy <= 70:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 275 and doy <= 340:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -2 and lat <= 2:
                if (doy >= 20 and doy <= 70) or (doy >= 190 and doy <= 250):
                    SOS_index.append(batch_df.index[0])
                elif (doy >= 90 and doy <= 150) or (doy >= 275 and doy <= 340):
                    EOS_index.append(batch_df.index[0])
            elif lat >= -6.5 and lat <= -2:
                if doy >= 190 and doy <= 250:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 80 and doy <= 150:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -9.5 and lat <= -6.5:
                if doy >= 180 and doy <= 250:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 60 and doy <= 150:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -12 and lat <= -9.5:
                if doy >= 190 and doy <= 275:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 50 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -16.5 and lat <= -12:
                if doy >= 200 and doy <= 275:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 50 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -18 and lat <= -16.5:
                if doy >= 220 and doy <= 300:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 40 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -20.75 and lat <= -18:
                if doy >= 240 and doy <= 320:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 40 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -22 and lat <= -21:
                if doy >= 200 and doy <= 275:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 40 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -22.75 and lat <= -22.25:
                if doy >= 240 and doy <= 320:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 40 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -37.75 and lat <= -23:
                if doy >= 200 and doy <= 320:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 30 and doy <= 110:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -41 and lat <= -38:
                if doy >= 200 and doy <= 300:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 0 and doy <= 100:
                    EOS_index.append(batch_df.index[0])
            elif lat >= -45.75 and lat <= -41:
                if doy >= 200 and doy <= 300:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 20 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
            elif lat <= -46:
                if doy >= 200 and doy <= 300:
                    SOS_index.append(batch_df.index[0])
                elif doy >= 30 and doy <= 120:
                    EOS_index.append(batch_df.index[0])
        else:
            # Default behavior: slope-based detection for non-BET PFTs
            x = range(len(batch_df))
            y = batch_df["CSIF"].values
            if abs(y[0] - y[-1]) > min_diff:
                slope, _, _, _, _ = linregress(x, y)
                if slope >= min_slope:
                    SOS_index.append(batch_df.index[0])
                elif slope <= -min_slope - 0.0005:
                    EOS_index.append(batch_df.index[0])

    SOS_indices = [int(i / 30) for i in SOS_index]
    EOS_indices = [int(i / 30) for i in EOS_index]

    # Filter out indices that are out of bounds for the predictions array
    max_pred_index = len(preds["attention_scores"]) - 1

    SOS_indices = [idx for idx in SOS_indices if idx <= max_pred_index]
    EOS_indices = [idx for idx in EOS_indices if idx <= max_pred_index]

    print(f"Number of valid SOS indices: {len(SOS_indices)}")
    print(f"Number of valid EOS indices: {len(EOS_indices)}")

    # attention drivers analysis
    save_imput_coord_att(
        SOS_indices,
        data,
        preds,
        coord_path,
        output_path + "_SOS.csv",
        forecast_window=args.forecast_window_length,
    )
    save_imput_coord_att(
        EOS_indices,
        data,
        preds,
        coord_path,
        output_path + "_EOS.csv",
        forecast_window=args.forecast_window_length,
    )

    # legacy analysis
    # legacy_df(SOS_indices, data, preds, coord_path, output_path + "_legacy_SOS.csv", size=0.25, forecast_window=args.forecast_window_length)
    # legacy_df(EOS_indices, data, preds, coord_path, output_path + "_legacy_EOS.csv", size=0.25, forecast_window=args.forecast_window_length)
