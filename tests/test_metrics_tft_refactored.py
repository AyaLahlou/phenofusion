"""
Unit tests for TFT metrics analysis module.

This module contains comprehensive tests for the refactored metrics analysis code.
"""

import unittest
import tempfile
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Import the classes to test
import sys

sys.path.append("/burg-archive/home/al4385/phenofusion/src/phenofusion/analysis")

from metrics_tft import (
    MetricsCalculator,
    DataProcessor,
    PlotConfiguration,
    MetricsPlotter,
)


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator class."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.observed = np.random.normal(0, 1, 100)
        self.predicted = self.observed + np.random.normal(0, 0.1, 100)

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = MetricsCalculator.calculate_metrics(self.observed, self.predicted)

        # Check that all expected metrics are present
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("bias", metrics)

        # Check that metrics are reasonable
        self.assertGreater(metrics["r2"], 0.8)  # Should be high correlation
        self.assertLess(metrics["rmse"], 0.5)  # Should be low RMSE
        self.assertLess(abs(metrics["bias"]), 0.2)  # Should be low bias

    def test_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        metrics = MetricsCalculator.calculate_metrics(self.observed, self.observed)

        self.assertAlmostEqual(metrics["r2"], 1.0, places=10)
        self.assertAlmostEqual(metrics["rmse"], 0.0, places=10)
        self.assertAlmostEqual(metrics["bias"], 0.0, places=10)


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""

    def setUp(self):
        """Set up test data and files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create mock data
        self.mock_prediction_data = {
            "predicted_quantiles": np.random.random((10, 5, 3))
        }

        self.mock_data = {
            "data_sets": {
                "test": {
                    "id": np.array(
                        [
                            f"loc_{i}_{pd.Timestamp('2020-01-01') + pd.Timedelta(days=j)}"
                            for i in range(2)
                            for j in range(25)
                        ]
                    ),
                    "target": np.random.random((10, 5)),
                }
            }
        }

        # Save mock data to files
        self.pred_file = self.temp_path / "predictions.pkl"
        self.data_file = self.temp_path / "data.pkl"

        with open(self.pred_file, "wb") as f:
            pickle.dump(self.mock_prediction_data, f)

        with open(self.data_file, "wb") as f:
            pickle.dump(self.mock_data, f)

    def test_load_pickle_file_success(self):
        """Test successful pickle file loading."""
        data = DataProcessor.load_pickle_file(self.pred_file)
        self.assertIn("predicted_quantiles", data)

    def test_load_pickle_file_not_found(self):
        """Test handling of missing files."""
        nonexistent_file = self.temp_path / "nonexistent.pkl"

        with self.assertRaises(FileNotFoundError):
            DataProcessor.load_pickle_file(nonexistent_file)

    def test_extract_predictions(self):
        """Test prediction extraction."""
        pred_01, pred_05, pred_09 = DataProcessor.extract_predictions(
            self.mock_prediction_data
        )

        self.assertEqual(pred_01.shape, (10, 5))
        self.assertEqual(pred_05.shape, (10, 5))
        self.assertEqual(pred_09.shape, (10, 5))

    def test_create_base_dataframe(self):
        """Test base dataframe creation."""
        predictions = DataProcessor.extract_predictions(self.mock_prediction_data)
        df = DataProcessor.create_base_dataframe(self.mock_data, predictions)

        # Check essential columns
        required_cols = ["observed", "pred_05", "location_id", "time_id", "doy"]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["time_id"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["location_id"]))


class TestPlotConfiguration(unittest.TestCase):
    """Test cases for PlotConfiguration class."""

    def test_colormap_creation(self):
        """Test colormap creation."""
        config = PlotConfiguration()
        colormap = config.create_colormap()

        self.assertIsNotNone(colormap)
        self.assertEqual(colormap.name, "white_viridis")

    def test_configuration_constants(self):
        """Test configuration constants."""
        config = PlotConfiguration()

        self.assertEqual(config.FIGURE_SIZE, (16, 4))
        self.assertEqual(config.SUBPLOT_COUNT, 4)
        self.assertIn("overall", config.SAMPLING_RATES)


class TestMetricsPlotter(unittest.TestCase):
    """Test cases for MetricsPlotter class."""

    def setUp(self):
        """Set up test data."""
        self.plotter = MetricsPlotter()

        # Create mock dataframe
        np.random.seed(42)
        n_points = 1000

        self.df = pd.DataFrame(
            {
                "observed": np.random.normal(0, 1, n_points),
                "pred_05": np.random.normal(0, 1, n_points),
                "temporal_var": np.random.normal(0, 0.5, n_points),
                "temporal_var_pred": np.random.normal(0, 0.5, n_points),
                "anomaly": np.random.normal(0, 0.1, n_points),
                "anomaly_pred": np.random.normal(0, 0.1, n_points),
            }
        )

        self.site_agg = pd.DataFrame(
            {
                "observed_site": np.random.normal(0, 1, 100),
                "predicted_site": np.random.normal(0, 1, 100),
            }
        )

    def test_calculate_plot_limits(self):
        """Test plot limits calculation."""
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        x_lim, y_lim = self.plotter._calculate_plot_limits(x_data, y_data)

        self.assertEqual(len(x_lim), 2)
        self.assertEqual(len(y_lim), 2)
        self.assertLess(x_lim[0], np.min(x_data))
        self.assertGreater(x_lim[1], np.max(x_data))

    def test_anomaly_plot_limits(self):
        """Test special handling of anomaly plot limits."""
        x_data = np.array([0.1, 0.2, 0.3])
        y_data = np.array([0.1, 0.2, 0.3])

        x_lim, y_lim = self.plotter._calculate_plot_limits(x_data, y_data, "anomaly")

        self.assertEqual(x_lim, (-0.05, 0.05))
        self.assertEqual(y_lim, (-0.05, 0.05))

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_metrics_plot(self, mock_close, mock_savefig):
        """Test complete metrics plot creation."""
        temp_dir = Path(tempfile.mkdtemp())
        output_path = temp_dir / "test_plot.png"

        metrics = self.plotter.create_metrics_plot(
            self.df, self.site_agg, "Test Plot", output_path
        )

        # Check that metrics were returned
        self.assertIn("r2", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("bias", metrics)

        # Check that each metric list has 4 values (one per subplot)
        self.assertEqual(len(metrics["r2"]), 4)
        self.assertEqual(len(metrics["rmse"]), 4)
        self.assertEqual(len(metrics["bias"]), 4)

        # Check that savefig was called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def setUp(self):
        """Set up integration test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create realistic mock data
        n_locations = 5
        n_timesteps = 100

        # Create prediction data
        prediction_data = {
            "predicted_quantiles": np.random.random((n_locations, n_timesteps, 3))
        }

        # Create observation data
        dates = pd.date_range("2020-01-01", periods=n_timesteps, freq="D")
        ids = []
        targets = []

        for loc in range(n_locations):
            for i, date in enumerate(dates):
                ids.append(f"{loc}_{date.strftime('%Y-%m-%d')}")
                targets.append(np.random.random())

        observation_data = {
            "data_sets": {
                "test": {
                    "id": np.array(ids),
                    "target": np.array(targets).reshape(n_locations, n_timesteps),
                }
            }
        }

        # Save to files
        pred_file = self.temp_path / "predictions.pkl"
        data_file = self.temp_path / "data.pkl"

        with open(pred_file, "wb") as f:
            pickle.dump(prediction_data, f)

        with open(data_file, "wb") as f:
            pickle.dump(observation_data, f)

        self.pred_file = pred_file
        self.data_file = data_file

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_full_pipeline(self, mock_close, mock_savefig):
        """Test the complete analysis pipeline."""
        output_path = self.temp_path / "output_plot.png"

        # Process data
        df, site_agg = DataProcessor.process_data(self.pred_file, self.data_file)

        # Create plots
        plotter = MetricsPlotter()
        metrics = plotter.create_metrics_plot(
            df, site_agg, "Integration Test", output_path
        )

        # Verify results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(site_agg, pd.DataFrame)
        self.assertIsInstance(metrics, dict)

        # Check that dataframe has expected structure
        expected_cols = [
            "observed",
            "pred_05",
            "location_id",
            "time_id",
            "doy",
            "observed_seasonal",
            "predicted_seasonal",
            "observed_site",
            "predicted_site",
            "temporal_var",
            "temporal_var_pred",
            "anomaly",
            "anomaly_pred",
        ]

        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")


if __name__ == "__main__":
    unittest.main()
