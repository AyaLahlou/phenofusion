"""
Configuration module for TFT metrics analysis.

This module contains configuration settings and constants used across
the metrics analysis pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class PlotConfig:
    """Configuration for plot styling and parameters."""

    # Figure dimensions
    figure_size: Tuple[int, int] = (16, 4)
    subplot_count: int = 4

    # Font sizes
    font_size_title: int = 16
    font_size_label: int = 16
    font_size_text: int = 15

    # Scatter plot parameters
    scatter_size: int = 10
    scatter_alpha: float = 0.8

    # Sampling rates for density plots (to improve performance)
    sampling_rates: Dict[str, int] = None

    # Axis limits for specific plot types
    axis_limits: Dict[str, Dict[str, Tuple[float, float]]] = None

    # Colormap parameters
    colormap_colors: Tuple[Tuple[float, str], ...] = (
        (0, "#ffffff"),  # Pure white at 0
        (1e-20, "#440053"),  # Dark purple
        (0.05, "#404388"),
        (0.2, "#2a788e"),  # Medium blue-green
        (0.4, "#21a784"),  # Teal
        (0.6, "#78d151"),  # Green
        (0.8, "#fde624"),  # Yellow
        (1, "#ffffcb"),  # Light yellow
    )

    def __post_init__(self):
        """Initialize default values for complex fields."""
        if self.sampling_rates is None:
            self.sampling_rates = {
                "overall": 1000,
                "site": 1000,
                "seasonal": 1000,
                "anomaly": 10,
            }

        if self.axis_limits is None:
            self.axis_limits = {"anomaly": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}}

    def create_colormap(self) -> LinearSegmentedColormap:
        """Create custom white-viridis colormap."""
        return LinearSegmentedColormap.from_list(
            "white_viridis",
            self.colormap_colors,
            N=256,
        )


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Column names mapping
    observed_col: str = "observed"
    predicted_col: str = "pred_05"
    location_col: str = "location_id"
    time_col: str = "time_id"
    doy_col: str = "doy"

    # Quantile indices in prediction arrays
    quantile_indices: Dict[str, int] = None

    # Data validation parameters
    min_data_points: int = 10
    max_missing_ratio: float = 0.1

    def __post_init__(self):
        """Initialize default values for complex fields."""
        if self.quantile_indices is None:
            self.quantile_indices = {
                "lower": 0,  # 0.1 quantile
                "median": 1,  # 0.5 quantile
                "upper": 2,  # 0.9 quantile
            }


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""

    # Metrics to calculate
    metrics_list: Tuple[str, ...] = ("rmse", "r2", "bias", "mae")

    # Precision for metrics display
    display_precision: int = 2

    # Validation thresholds
    min_r2: float = 0.0
    max_rmse_ratio: float = 2.0  # Max RMSE as ratio of data range


# Default configuration instances
DEFAULT_PLOT_CONFIG = PlotConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_METRICS_CONFIG = MetricsConfig()
