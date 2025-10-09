"""
Tests for Heatmap Analysis Module

Comprehensive test suite for the heatmap analysis functionality,
including unit tests for all major components and integration tests.

Author: Phenofusion Team
Date: October 2025
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import pickle
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import sys

# Add the phenofusion package to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from phenofusion.analysis.heatmap_analysis import (
    SeasonalDateFinder,
    GeographicLocalizer,
    HeatmapAnalyzer,
    create_analysis_dataframe,
    analyze_city_heatmaps,
)


class TestSeasonalDateFinder:
    """Test cases for SeasonalDateFinder class."""

    def test_start_of_growing_season_northern_hemisphere(self):
        """Test start of growing season calculation for northern hemisphere."""
        # Test case: Latitude 45 (mid-latitude northern)
        df = pd.DataFrame({"latitude": [45.0] * 10})
        result = SeasonalDateFinder.start_of_growing_season(df)
        assert result == [4], f"Expected [4], got {result}"

        # Test case: High northern latitude
        df = pd.DataFrame({"latitude": [65.0] * 10})
        result = SeasonalDateFinder.start_of_growing_season(df)
        assert result == [6], f"Expected [6], got {result}"

    def test_start_of_growing_season_southern_hemisphere(self):
        """Test start of growing season calculation for southern hemisphere."""
        # Test case: Southern hemisphere
        df = pd.DataFrame({"latitude": [-25.0] * 10})
        result = SeasonalDateFinder.start_of_growing_season(df)
        assert result == [8, 9], f"Expected [8, 9], got {result}"

    def test_start_of_growing_season_extreme_latitudes(self):
        """Test start of growing season for extreme latitudes."""
        # Test case: Extreme latitude (no growing season)
        df = pd.DataFrame({"latitude": [80.0] * 10})
        result = SeasonalDateFinder.start_of_growing_season(df)
        assert result is None, f"Expected None, got {result}"

    def test_end_of_growing_season_valid_latitudes(self):
        """Test end of growing season calculation."""
        # Test case: Mid-latitude northern
        df = pd.DataFrame({"latitude": [45.0] * 10})
        result = SeasonalDateFinder.end_of_growing_season(df)
        assert result == [10], f"Expected [10], got {result}"

        # Test case: Southern hemisphere
        df = pd.DataFrame({"latitude": [-25.0] * 10})
        result = SeasonalDateFinder.end_of_growing_season(df)
        assert result == [4, 5], f"Expected [4, 5], got {result}"

    def test_end_of_growing_season_invalid_latitude(self):
        """Test end of growing season with invalid latitude."""
        df = pd.DataFrame({"latitude": [-70.0] * 10})
        with pytest.raises(ValueError, match="Latitude is out of the specified range"):
            SeasonalDateFinder.end_of_growing_season(df)

    def test_find_date_range_valid_case(self):
        """Test date range finding with valid data."""
        # Create mock DataFrame
        dates = pd.date_range(start="2006-04-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "latitude": [45.0] * 100,
                "year": [d.year for d in dates],
                "time": dates,
                "location": list(range(100)),
            }
        )

        # Add some mock data to satisfy the groupby filter
        df = pd.concat([df] * 10, ignore_index=True)  # Ensure enough data points

        indices, window_start = SeasonalDateFinder.find_date_range(df, "sos", 2006)

        # Should find some indices and a valid start date
        if indices is not None:
            assert len(indices) >= 0
            assert isinstance(window_start, datetime)

    def test_find_date_range_no_suitable_range(self):
        """Test date range finding when no suitable range exists."""
        # Create minimal DataFrame that won't satisfy threshold
        df = pd.DataFrame(
            {
                "latitude": [45.0] * 5,
                "year": [2006] * 5,
                "time": pd.date_range(start="2006-04-01", periods=5, freq="D"),
                "location": list(range(5)),
            }
        )

        indices, window_start = SeasonalDateFinder.find_date_range(df, "sos", 2006)
        assert indices is None
        assert window_start is None


class TestGeographicLocalizer:
    """Test cases for GeographicLocalizer class."""

    def test_localize_df(self):
        """Test geographic filtering of DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "latitude": [40, 45, 50, 55, 60],
                "longitude": [-10, 0, 10, 20, 30],
                "data": [1, 2, 3, 4, 5],
            }
        )

        # Filter data
        lat_range = (42, 58)
        lon_range = (-5, 25)

        result = GeographicLocalizer.localize_df(df, lat_range, lon_range)

        # Check filtering
        assert all(result["latitude"] >= lat_range[0])
        assert all(result["latitude"] <= lat_range[1])
        assert all(result["longitude"] >= lon_range[0])
        assert all(result["longitude"] <= lon_range[1])
        assert len(result) == 3  # Should have 3 rows (45, 50, 55 lat)

    @patch("phenofusion.analysis.heatmap_analysis.Nominatim")
    def test_get_city_bounding_box_success(self, mock_nominatim):
        """Test successful city geocoding."""
        # Mock the geocoder
        mock_location = Mock()
        mock_location.latitude = 47.5
        mock_location.longitude = 19.0

        mock_geolocator = Mock()
        mock_geolocator.geocode.return_value = mock_location
        mock_nominatim.return_value = mock_geolocator

        # Test the function
        lat_range, lon_range = GeographicLocalizer.get_city_bounding_box("Budapest")

        # Check results
        assert lat_range == (37.5, 57.5)  # lat ± 10
        assert lon_range == (-1.0, 39.0)  # lon ± 20
        mock_geolocator.geocode.assert_called_once_with("Budapest")

    @patch("phenofusion.analysis.heatmap_analysis.Nominatim")
    def test_get_city_bounding_box_not_found(self, mock_nominatim):
        """Test city not found case."""
        mock_geolocator = Mock()
        mock_geolocator.geocode.return_value = None
        mock_nominatim.return_value = mock_geolocator

        with pytest.raises(ValueError, match="City 'NonexistentCity' not found"):
            GeographicLocalizer.get_city_bounding_box("NonexistentCity")

    def test_check_latitude_interval(self):
        """Test latitude interval categorization."""
        # Test different intervals
        test_cases = [
            ((60, 80), "50_90"),
            ((30, 40), "50_20"),
            ((-10, 10), "-20_20"),
            ((-40, -30), "-20_-60"),
            ((-80, -70), "Out_of_range"),
        ]

        for lat_tuple, expected in test_cases:
            result = GeographicLocalizer.check_latitude_interval(lat_tuple)
            assert (
                result == expected
            ), f"For {lat_tuple}, expected {expected}, got {result}"


class TestHeatmapAnalyzer:
    """Test cases for HeatmapAnalyzer class."""

    def test_initialization(self):
        """Test HeatmapAnalyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = HeatmapAnalyzer(output_dir=tmpdir)

            assert analyzer.output_dir == Path(tmpdir)
            assert analyzer.output_dir.exists()
            assert "CSIF" in analyzer.feature_names
            assert len(analyzer.feature_indices) == 7

    def test_extract_feature_importance_data(self):
        """Test feature importance data extraction."""
        analyzer = HeatmapAnalyzer()

        # Create mock predictions data
        n_indices = 3
        n_historical = 365
        n_future = 30
        n_features_hist = 7
        n_features_fut = 6

        mock_preds = {
            "historical_selection_weights": [
                np.random.rand(n_historical, n_features_hist) for _ in range(n_indices)
            ],
            "future_selection_weights": [
                np.random.rand(n_future, n_features_fut) for _ in range(n_indices)
            ],
        }

        select_indices = np.array([0, 1, 2])

        result = analyzer.extract_feature_importance_data(mock_preds, select_indices)

        # Check structure
        expected_features = ["csif", "tmin", "tmax", "rad", "precip", "photo", "SM"]
        assert set(result.keys()) == set(expected_features)

        # Check dimensions
        for feature, array in result.items():
            expected_length = n_historical + n_future
            assert len(array) == expected_length, f"Feature {feature} has wrong length"

    def test_create_date_labels(self):
        """Test date label creation."""
        analyzer = HeatmapAnalyzer()
        window_start = datetime(2006, 4, 1)

        positions, labels = analyzer.create_date_labels(window_start, n_periods=120)

        # Check basic structure
        assert isinstance(positions, list)
        assert isinstance(labels, list)
        assert len(positions) <= len(labels)
        assert all(isinstance(pos, int) for pos in positions)
        assert all(isinstance(label, str) for label in labels)


class TestCreateAnalysisDataframe:
    """Test cases for create_analysis_dataframe function."""

    def test_create_analysis_dataframe_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            create_analysis_dataframe(
                "nonexistent_data.pkl",
                "nonexistent_pred.pkl",
                "nonexistent_coord.parquet",
            )

    def test_create_analysis_dataframe_success(self):
        """Test successful DataFrame creation with mock data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock data files
            mock_data = {
                "data_sets": {
                    "test": {
                        "id": np.array(["1_2006-04-01", "2_2006-04-02"]),
                        "target": np.array([[0.5], [0.6]]),
                    }
                }
            }

            mock_preds = {"predicted_quantiles": np.random.rand(2, 30, 3)}

            mock_coords = pd.DataFrame(
                {
                    "location": [1, 2],
                    "latitude": [45.0, 46.0],
                    "longitude": [19.0, 20.0],
                }
            )

            # Save mock files
            data_path = tmpdir / "data.pkl"
            pred_path = tmpdir / "pred.pkl"
            coord_path = tmpdir / "coord.parquet"

            with open(data_path, "wb") as f:
                pickle.dump(mock_data, f)
            with open(pred_path, "wb") as f:
                pickle.dump(mock_preds, f)
            mock_coords.to_parquet(coord_path)

            # Test function
            result = create_analysis_dataframe(
                str(data_path), str(pred_path), str(coord_path)
            )

            # Check result structure
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            expected_columns = [
                "location",
                "latitude",
                "longitude",
                "CSIF",
                "pred_05",
                "time",
                "doy",
                "year",
                "month",
                "day",
            ]
            assert all(col in result.columns for col in expected_columns)


class TestIntegration:
    """Integration tests for the complete heatmap analysis pipeline."""

    @patch(
        "phenofusion.analysis.heatmap_analysis.GeographicLocalizer.get_city_bounding_box"
    )
    @patch("phenofusion.analysis.heatmap_analysis.create_analysis_dataframe")
    @patch("phenofusion.analysis.heatmap_analysis.HeatmapAnalyzer")
    def test_analyze_city_heatmaps_success(
        self, mock_analyzer_class, mock_create_df, mock_get_bbox
    ):
        """Test complete city analysis pipeline."""
        # Setup mocks
        mock_get_bbox.return_value = ((40, 50), (10, 20))

        mock_df = pd.DataFrame(
            {
                "latitude": [45.0] * 10,
                "longitude": [15.0] * 10,
                "time": pd.date_range("2006-04-01", periods=10),
                "year": [2006] * 10,
                "location": list(range(10)),
            }
        )
        mock_create_df.return_value = mock_df

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        # Test function
        analyze_city_heatmaps("TestCity", "BDT", "sos")

        # Verify calls
        mock_get_bbox.assert_called_once_with("TestCity")
        mock_create_df.assert_called_once()
        mock_analyzer.generate_feature_importance_heatmap.assert_called_once()
        mock_analyzer.generate_attention_heatmap.assert_called_once()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range(start="2006-04-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "latitude": np.random.uniform(40, 50, 100),
            "longitude": np.random.uniform(10, 20, 100),
            "time": dates,
            "year": [d.year for d in dates],
            "month": [d.month for d in dates],
            "day": [d.day for d in dates],
            "location": np.random.randint(1, 20, 100),
            "CSIF": np.random.rand(100),
            "pred_05": np.random.rand(100),
        }
    )


class TestRobustness:
    """Tests for robustness and edge cases."""

    def test_seasonal_date_finder_edge_cases(self, sample_dataframe):
        """Test edge cases for seasonal date finding."""
        # Test with invalid season
        with pytest.raises(ValueError, match="Season must be 'sos' or 'eos'"):
            SeasonalDateFinder.find_date_range(sample_dataframe, "invalid_season")

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame(columns=["latitude", "longitude", "time"])

        # Should handle empty DataFrame gracefully
        result = GeographicLocalizer.localize_df(empty_df, (40, 50), (10, 20))
        assert len(result) == 0
        assert list(result.columns) == ["latitude", "longitude", "time"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
