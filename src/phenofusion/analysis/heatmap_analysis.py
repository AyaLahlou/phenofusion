"""
Heatmap Analysis Module for Phenological Feature Importance and Attention Visualization

This module provides comprehensive tools for analyzing temporal feature importance and attention
patterns in phenological forecasting models. It generates publication-ready heatmaps combining
attention scores with time series data for climate driver analysis.

Classes:
    HeatmapAnalyzer: Main class for generating feature importance and attention heatmaps
    SeasonalDateFinder: Utility class for determining growing season dates
    GeographicLocalizer: Utility class for geographic data filtering

Functions:
    create_analysis_dataframe: Load and merge prediction data with coordinates
    generate_feature_importance_heatmap: Create feature importance visualization
    generate_attention_heatmap: Create attention pattern visualization

Author: Phenofusion Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeasonalDateFinder:
    """Utility class for determining growing season dates based on latitude."""

    @staticmethod
    def start_of_growing_season(df: pd.DataFrame) -> Optional[List[int]]:
        """
        Determine start of growing season months based on median latitude.

        Args:
            df: DataFrame containing latitude data

        Returns:
            List of month numbers for growing season start, or None if not defined
        """
        latitude = df["latitude"].median()
        logger.info(f"Median latitude: {latitude}")

        if latitude > 70 or latitude < -70:
            return None  # Growing season not defined for these latitudes

        # Northern Hemisphere
        if 56 <= latitude <= 70:
            return [6]
        elif 50 <= latitude < 56:
            return [4, 5]
        elif 40 <= latitude < 50:
            return [4]
        elif 30 <= latitude < 40:
            return [2, 3]
        elif 10 <= latitude < 30:
            return [3]
        elif 0 <= latitude < 10:
            return [3]

        # Southern Hemisphere
        if -10 <= latitude < 0:
            return [8]
        elif -30 <= latitude < -10:
            return [8, 9]
        elif -40 <= latitude < -30:
            return [8, 9]
        elif -50 <= latitude < -40:
            return [9, 10]
        elif -60 <= latitude < -50:
            return [9, 10]
        elif -70 <= latitude < -60:
            return [12, 1]

        return None

    @staticmethod
    def end_of_growing_season(df: pd.DataFrame) -> List[int]:
        """
        Determine end of growing season months based on median latitude.

        Args:
            df: DataFrame containing latitude data

        Returns:
            List of month numbers for growing season end

        Raises:
            ValueError: If latitude is out of specified range
        """
        latitude = df["latitude"].median()

        if 90 >= latitude > 60:
            return [9]
        elif 60 >= latitude > 50:
            return [9, 10]
        elif 50 >= latitude > 20:
            return [10]
        elif 20 >= latitude > 10:
            return [10, 11]
        elif 10 >= latitude > 0:
            return [12]
        elif 0 >= latitude > -12:
            return [7]
        elif -12 >= latitude > -20:
            return [5]
        elif -20 >= latitude > -30:
            return [4, 5]
        elif -30 >= latitude > -60:
            return [4]
        else:
            raise ValueError("Latitude is out of the specified range")

    @staticmethod
    def find_date_range(
        df: pd.DataFrame, season: str, specific_year: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[datetime]]:
        """
        Find optimal date range with sufficient sites for analysis.

        This function iterates through different date windows within growing season
        months to find a start date with at least 7 sites sharing the same start date.

        Args:
            df: DataFrame with time series data
            season: Either 'eos' (end of season) or 'sos' (start of season)
            specific_year: Optional specific year to analyze

        Returns:
            Tuple of (selected indices array, window start date) or (None, None)
        """
        if season == "eos":
            months = SeasonalDateFinder.end_of_growing_season(df)
        elif season == "sos":
            months = SeasonalDateFinder.start_of_growing_season(df)
        else:
            raise ValueError("Season must be 'sos' or 'eos'")

        if months is None or len(months) > 2:
            logger.warning("Equator median latitude, no SOS/EOS Date range")
            return None, None

        # Determine date window based on months
        if len(months) == 2:
            start_month, start_day = months[0], 15
            end_month, end_day = months[1], 15
        else:
            start_month, start_day = months[0], 1
            end_month = months[0] + 1 if months[0] != 12 else 1
            end_day = 1

        # Analyze years
        years = [specific_year] if specific_year else range(2002, 2011)

        for year in years:
            dt_start = datetime(year, start_month, start_day)

            # Handle year transitions
            if start_month != end_month and end_month == 1:
                year_2 = year + 1
                dt_end = datetime(year_2, end_month, end_day)
                df_current_year = df[(df["year"] >= year) & (df["year"] <= year_2)]
            else:
                dt_end = datetime(year, end_month, end_day)
                df_current_year = df[df["year"] == year]

            # Sliding window analysis
            counter = 0
            index_threshold = 7

            while True:
                window_start = dt_start + timedelta(days=counter)
                window_end = dt_start + timedelta(days=counter + 3)

                if window_end >= dt_end:
                    break

                filtered_df = df_current_year.groupby("location").filter(
                    lambda x: (x["time"].iloc[0] >= window_start)
                    and (x["time"].iloc[0] <= window_end)
                )

                indices_array = np.unique(filtered_df[::30].index.to_numpy())
                select_indices = np.round(indices_array / 30).astype(int)

                if len(select_indices) > index_threshold:
                    logger.info(
                        f"Found suitable window: {window_start} to {window_end}"
                    )
                    return select_indices, window_start

                counter += 1

        logger.warning("No suitable date range found")
        return None, None


class GeographicLocalizer:
    """Utility class for geographic data filtering and location services."""

    @staticmethod
    def localize_df(
        df: pd.DataFrame,
        latitude_range: Tuple[float, float],
        longitude_range: Tuple[float, float],
    ) -> pd.DataFrame:
        """
        Filter DataFrame by geographic bounds.

        Args:
            df: Input DataFrame with latitude/longitude columns
            latitude_range: Tuple of (min_lat, max_lat)
            longitude_range: Tuple of (min_lon, max_lon)

        Returns:
            Filtered DataFrame
        """
        df_filtered = df[
            (df["latitude"] > latitude_range[0])
            & (df["latitude"] < latitude_range[1])
            & (df["longitude"] > longitude_range[0])
            & (df["longitude"] < longitude_range[1])
        ]

        logger.info(
            f"Latitude range: {df_filtered['latitude'].min():.2f} to {df_filtered['latitude'].max():.2f}"
        )
        logger.info(f"Median latitude: {df_filtered['latitude'].median():.2f}")
        logger.info(
            f"Longitude range: {df_filtered['longitude'].min():.2f} to {df_filtered['longitude'].max():.2f}"
        )

        return df_filtered

    @staticmethod
    def get_city_bounding_box(
        city_name: str,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get bounding box coordinates for a city.

        Args:
            city_name: Name of the city

        Returns:
            Tuple of ((min_lat, max_lat), (min_lon, max_lon))

        Raises:
            ValueError: If city not found
        """
        try:
            from geopy.geocoders import Nominatim
        except ImportError:
            raise ImportError(
                "geopy is required for city geocoding. Install with: pip install geopy"
            )

        geolocator = Nominatim(user_agent="phenofusion_heatmap_analysis")
        location = geolocator.geocode(city_name)

        if location is None:
            raise ValueError(f"City '{city_name}' not found.")

        lat, lon = location.latitude, location.longitude

        # Create bounding box (adjustable based on analysis needs)
        lat_range = (lat - 10, lat + 10)
        lon_range = (lon - 20, lon + 20)

        return lat_range, lon_range

    @staticmethod
    def check_latitude_interval(lat_tuple: Tuple[float, float]) -> str:
        """
        Determine latitude interval category for data file selection.

        Args:
            lat_tuple: Tuple of (min_lat, max_lat)

        Returns:
            String representing latitude interval category
        """
        mean_lat = (lat_tuple[0] + lat_tuple[1]) / 2

        if 50 <= mean_lat <= 90:
            return "50_90"
        elif 20 <= mean_lat < 50:
            return "50_20"
        elif -20 <= mean_lat < 20:
            return "-20_20"
        elif -60 <= mean_lat < -20:
            return "-20_-60"
        else:
            return "Out_of_range"


def create_analysis_dataframe(
    data_path: str, pred_path: str, coord_path: str
) -> pd.DataFrame:
    """
    Load and merge prediction data with coordinates for analysis.

    Args:
        data_path: Path to original processed data dictionary
        pred_path: Path to predictions dictionary
        coord_path: Path to coordinates DataFrame

    Returns:
        Merged DataFrame with predictions and coordinates

    Raises:
        FileNotFoundError: If any input file doesn't exist
        KeyError: If expected data structure is missing
    """
    # Validate file paths
    for path in [data_path, pred_path, coord_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    try:
        # Load data files
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)
        with open(pred_path, "rb") as fp:
            preds = pickle.load(fp)

        coords = pd.read_parquet(coord_path).drop_duplicates()

        # Create base DataFrame
        df = pd.DataFrame(
            {
                "Index": data["data_sets"]["test"]["id"].flatten(),
                "CSIF": data["data_sets"]["test"]["target"].flatten(),
            }
        )

        # Add predictions
        df["pred_05"] = preds["predicted_quantiles"][:, :, 1].flatten()

        # Parse location and time information
        df[["location", "time"]] = df["Index"].str.split("_", n=1, expand=True)
        df["location"] = df["location"].astype(int)
        df["time"] = pd.to_datetime(df["time"])

        # Add temporal features
        df["doy"] = df["time"].dt.dayofyear
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day

        # Merge with coordinates
        df = pd.merge(coords, df.drop(columns=["Index"]), on="location", how="left")
        df = df.sort_values(by=["location", "time"])

        logger.info(f"Created analysis DataFrame with {len(df)} records")
        return df

    except Exception as e:
        logger.error(f"Error creating analysis DataFrame: {e}")
        raise


class HeatmapAnalyzer:
    """
    Main class for generating feature importance and attention heatmaps.

    This class provides methods to create publication-ready visualizations
    of temporal feature importance and attention patterns in phenological models.
    """

    def __init__(self, output_dir: str = "feature_importance_plots"):
        """
        Initialize HeatmapAnalyzer.

        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define feature names and colors
        self.feature_names = {
            "CSIF": "CSIF",
            "Tmin": "Temperature Min",
            "Tmax": "Temperature Max",
            "SR": "Solar Radiation",
            "PR": "Precipitation",
            "PP": "Photoperiod",
            "SM": "Soil Moisture",
        }

        self.feature_indices = {
            "csif": 0,
            "tmin": 1,
            "tmax": 2,
            "rad": 3,
            "precip": 4,
            "photo": 5,
            "SM": 6,
        }

    def extract_feature_importance_data(
        self, preds: Dict, select_indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract feature importance weights from predictions.

        Args:
            preds: Predictions dictionary containing selection weights
            select_indices: Array of selected time series indices

        Returns:
            Dictionary containing mean feature importance arrays
        """
        # Initialize lists for each feature
        feature_lists = {
            "csif": [],
            "tmin": [],
            "tmax": [],
            "rad": [],
            "precip": [],
            "photo": [],
            "SM": [],
        }

        # Extract data for each selected index
        for index in select_indices:
            # CSIF (historical only, padded with zeros for future)
            feature_lists["csif"].append(
                np.concatenate(
                    [preds["historical_selection_weights"][index][:, 0], np.zeros(30)]
                )
            )

            # Other features (historical + future)
            for i, feature in enumerate(
                ["tmin", "tmax", "rad", "precip", "photo", "SM"]
            ):
                hist_idx = i + 1  # Skip CSIF in historical
                fut_idx = i  # Direct mapping in future

                feature_lists[feature].append(
                    np.concatenate(
                        [
                            preds["historical_selection_weights"][index][:, hist_idx],
                            preds["future_selection_weights"][index][:, fut_idx],
                        ]
                    )
                )

        # Calculate means
        return {
            feature: np.mean(np.stack(arrays, axis=0), axis=0)
            for feature, arrays in feature_lists.items()
        }

    def extract_timeseries_data(
        self, data: Dict, preds: Dict, select_indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract and normalize time series data.

        Args:
            data: Original data dictionary
            preds: Predictions dictionary
            select_indices: Array of selected time series indices

        Returns:
            Dictionary containing normalized time series arrays
        """
        # Initialize lists
        series_lists = {
            "csif_obs": [],
            "csif_pred": [],
            "tmin": [],
            "tmax": [],
            "rad": [],
            "precip": [],
            "photo": [],
            "SM": [],
        }

        # Extract data for each index
        for index in select_indices:
            # Observed and predicted CSIF
            series_lists["csif_obs"].append(
                np.concatenate(
                    [
                        data["data_sets"]["test"]["historical_ts_numeric"][index][:, 0],
                        data["data_sets"]["test"]["target"][index],
                    ]
                )
            )
            series_lists["csif_pred"].append(
                preds["predicted_quantiles"][:, :, 1][index].flatten()
            )

            # Other variables
            for i, var in enumerate(["tmin", "tmax", "rad", "precip", "photo", "SM"]):
                series_lists[var].append(
                    np.concatenate(
                        [
                            data["data_sets"]["test"]["historical_ts_numeric"][index][
                                :, i + 1
                            ],
                            data["data_sets"]["test"]["future_ts_numeric"][index][:, i],
                        ]
                    )
                )

        # Calculate means and normalize CSIF
        result = {}
        csif_obs = np.mean(np.stack(series_lists["csif_obs"], axis=0), axis=0)
        csif_pred = np.mean(np.stack(series_lists["csif_pred"], axis=0), axis=0)

        # Normalize CSIF using combined min/max
        combined = np.concatenate([csif_obs, csif_pred])
        min_val, max_val = combined.min(), combined.max()

        result["CSIF"] = (csif_obs - min_val) / (max_val - min_val)
        result["CSIF_pred"] = (csif_pred - min_val) / (max_val - min_val)

        # Add other variables (no normalization)
        for var in ["tmin", "tmax", "rad", "precip", "photo", "SM"]:
            result[
                (
                    var.upper()
                    if var in ["tmin", "tmax", "rad", "precip", "photo"]
                    else var
                )
            ] = np.mean(np.stack(series_lists[var], axis=0), axis=0)

        return result

    def create_date_labels(
        self, window_start: datetime, n_periods: int = 395
    ) -> Tuple[List[int], List[str]]:
        """
        Create date labels for x-axis.

        Args:
            window_start: Starting date for the time series
            n_periods: Number of time periods

        Returns:
            Tuple of (first day positions, month labels)
        """
        date_range = pd.date_range(start=window_start, periods=n_periods, freq="D")
        df_dates = pd.DataFrame({"Date": date_range, "Index": range(n_periods)})

        first_day_positions = df_dates[df_dates["Date"].dt.is_month_start][
            "Index"
        ].tolist()

        month_range = pd.date_range(
            start=window_start, periods=len(first_day_positions) + 1, freq="M"
        )
        month_labels = [date.strftime("%b") for date in month_range][1:]

        return first_day_positions, month_labels

    def generate_feature_importance_heatmap(
        self,
        df: pd.DataFrame,
        pred_path: str,
        data_path: str,
        season: str,
        specific_year: Optional[int] = None,
        title: str = "",
    ) -> None:
        """
        Generate comprehensive feature importance heatmap.

        Args:
            df: Analysis DataFrame with location and time data
            pred_path: Path to predictions file
            data_path: Path to original data file
            season: Season type ('sos' or 'eos')
            specific_year: Optional specific year to analyze
            title: Plot title
        """
        # Load prediction and data files
        with open(pred_path, "rb") as fp:
            preds = pickle.load(fp)
        with open(data_path, "rb") as fp:
            data = pickle.load(fp)

        # Find optimal date range
        select_indices, window_start = SeasonalDateFinder.find_date_range(
            df, season, specific_year
        )

        if select_indices is None:
            logger.error("No suitable date range found for analysis")
            return

        logger.info(f"Using {len(select_indices)} sites for analysis")

        # Extract feature importance and time series data
        feature_importance = self.extract_feature_importance_data(preds, select_indices)
        timeseries_data = self.extract_timeseries_data(data, preds, select_indices)

        # Create plot
        self._plot_feature_heatmap(
            feature_importance, timeseries_data, window_start, title
        )

    def _plot_feature_heatmap(
        self,
        feature_importance: Dict[str, np.ndarray],
        timeseries_data: Dict[str, np.ndarray],
        window_start: datetime,
        title: str,
    ) -> None:
        """
        Create the actual heatmap plot.

        Args:
            feature_importance: Dictionary of feature importance arrays
            timeseries_data: Dictionary of time series arrays
            window_start: Start date for x-axis labels
            title: Plot title
        """
        # Setup figure
        fig, axes = plt.subplots(8, 1, figsize=(17, 12))
        fig.suptitle(
            f"{title} Feature Importance - {window_start.strftime('%Y-%m-%d')}",
            fontsize=14,
            y=0.98,
        )

        # Create date labels
        first_day_positions, month_labels = self.create_date_labels(window_start)

        # Prepare feature data
        feature_df = pd.DataFrame(
            {
                name.upper(): feature_importance[name.lower()]
                for name in ["tmin", "tmax", "rad", "precip", "photo", "SM"]
            }
        )

        # Sort features by maximum importance
        sorted_features = feature_df.max().sort_values(ascending=False).index.tolist()

        # Plot CSIF first (special case)
        ax = axes[0]
        csif_heatmap = pd.DataFrame(
            {"CSIF": np.zeros(len(feature_importance["csif"]))}
        ).T

        sns.heatmap(
            csif_heatmap,
            cmap="gnuplot2_r",
            vmax=0.7,
            cbar=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.plot(timeseries_data["CSIF"], color="blue", linewidth=2, label="Observed")
        ax.plot(
            np.arange(366, 396),
            timeseries_data["CSIF_pred"],
            color="red",
            linewidth=2,
            label="Predicted",
        )
        ax.axvline(x=365, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("CSIF", rotation=0, ha="right")
        ax.set_xticks([])
        ax.legend(loc="upper right", fontsize=8)

        # Plot other features
        for i, feature in enumerate(sorted_features, 1):
            ax = axes[i]

            # Heatmap
            heatmap_data = feature_df[[feature]].T
            sns.heatmap(
                heatmap_data,
                cmap="gnuplot2_r",
                vmax=0.7,
                cbar=True,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )

            # Time series overlay
            ts_key = feature if feature in timeseries_data else feature.upper()
            if ts_key in timeseries_data:
                ax.plot(timeseries_data[ts_key], color="blue", linewidth=2)

            ax.axvline(x=365, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylabel(feature, rotation=0, ha="right")

            if i < len(sorted_features):
                ax.set_xticks([])

        # Final axis formatting
        axes[-1].set_xticks(first_day_positions)
        axes[-1].set_xticklabels(month_labels, ha="center")
        axes[-1].set_xlabel("Time", fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        # Save plot
        output_path = (
            self.output_dir / f"feature_importance_{title.replace(' ', '_')}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {output_path}")
        plt.show()

    def generate_attention_heatmap(
        self,
        df: pd.DataFrame,
        pred_path: str,
        season: str,
        specific_year: Optional[int] = None,
        title: str = "",
    ) -> None:
        """
        Generate attention pattern heatmap.

        Args:
            df: Analysis DataFrame
            pred_path: Path to predictions file
            season: Season type ('sos' or 'eos')
            specific_year: Optional specific year
            title: Plot title
        """
        with open(pred_path, "rb") as fp:
            preds = pickle.load(fp)

        select_indices, window_start = SeasonalDateFinder.find_date_range(
            df, season, specific_year
        )

        if select_indices is None:
            logger.error("No suitable date range found for attention analysis")
            return

        # Extract attention matrices
        attention_matrices = [preds["attention_scores"][idx] for idx in select_indices]
        mean_attention = np.mean(np.stack(attention_matrices, axis=0), axis=0)

        # Create plot
        plt.figure(figsize=(20, 6))

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list("attention", ["white", "red"])

        sns.heatmap(mean_attention, cmap=cmap, cbar_kws={"shrink": 0.8})
        plt.axvline(x=365, color="black", linestyle="--", linewidth=2, alpha=0.8)

        # Labels and formatting
        first_day_positions, month_labels = self.create_date_labels(window_start)
        plt.xticks(first_day_positions, month_labels, ha="center")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Forecast Horizon", fontsize=12)
        plt.title(
            f"{title} Attention Patterns - {window_start.strftime('%Y-%m-%d')}",
            fontsize=14,
        )

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f"attention_{title.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved attention plot to {output_path}")
        plt.show()


def analyze_city_heatmaps(
    city: str,
    pft: str = "BDT",
    season: str = "sos",
    specific_year: Optional[int] = None,
    base_data_dir: str = "/burg/glab/users/al4385/data/TFT_30",
    base_pred_dir: str = "/burg/glab/users/al4385/predictions/TFT_30_0429",
    base_coord_dir: str = "/burg/glab/users/al4385/data/coordinates",
) -> None:
    """
    Complete analysis pipeline for a city.

    Args:
        city: Name of the city to analyze
        pft: Plant functional type (e.g., 'BDT', 'BET')
        season: Season type ('sos' or 'eos')
        specific_year: Optional specific year to analyze
        base_data_dir: Base directory for data files
        base_pred_dir: Base directory for prediction files
        base_coord_dir: Base directory for coordinate files
    """
    try:
        # Get geographic bounds
        lat_range, lon_range = GeographicLocalizer.get_city_bounding_box(city)

        # Determine file paths
        if pft == "BDT":
            lat_interval = GeographicLocalizer.check_latitude_interval(lat_range)
            file_suffix = f"{pft}_{lat_interval}"
        else:
            file_suffix = pft

        data_path = f"{base_data_dir}/{file_suffix}.pickle"
        pred_path = f"{base_pred_dir}/pred_{file_suffix}.pkl"
        coord_path = f"{base_coord_dir}/{file_suffix}.parquet"

        # Create analysis DataFrame
        df = create_analysis_dataframe(data_path, pred_path, coord_path)
        df = GeographicLocalizer.localize_df(df, lat_range, lon_range)

        # Initialize analyzer
        analyzer = HeatmapAnalyzer()

        # Generate visualizations
        analyzer.generate_feature_importance_heatmap(
            df, pred_path, data_path, season, specific_year, title=city
        )
        analyzer.generate_attention_heatmap(
            df, pred_path, season, specific_year, title=city
        )

        logger.info(f"Completed analysis for {city}")

    except Exception as e:
        logger.error(f"Error analyzing {city}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    analyze_city_heatmaps("Budapest", pft="BDT", season="sos")
