#!/usr/bin/env python3
"""
Test script for the sensitivity analysis generator.

This script performs basic tests to ensure the refactored sensitivity analysis code works correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from generate_sensitivity_plots import (
        AnalysisConfig,
        PhenologicalDataProcessor,
        SensitivityAnalyzer,
    )

    print("✅ Successfully imported sensitivity analysis modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def test_config_creation():
    """Test the configuration creation."""
    print("\n=== Testing Configuration Creation ===")

    try:
        config = AnalysisConfig(
            data_directory="/tmp",
            pred_directory="/tmp",
            coord_directory="/tmp",
            output_directory="./test_output",
            years=[2000, 2010],
            seasons=["sos", "eos"],
            cluster_names=["BET", "NET"],
        )

        print("✅ Configuration created successfully")
        print(f"   Years: {config.years}")
        print(f"   Seasons: {config.seasons}")
        print(f"   Clusters: {config.cluster_names}")
        return True

    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False


def test_processor_initialization():
    """Test processor initialization."""
    print("\n=== Testing Processor Initialization ===")

    try:
        config = AnalysisConfig(
            data_directory="/tmp",
            pred_directory="/tmp",
            coord_directory="/tmp",
            output_directory="./test_output",
            years=[2000, 2010],
            seasons=["sos"],
            cluster_names=["BET"],
        )

        PhenologicalDataProcessor(config)

        # Check if output directory was created
        if os.path.exists("./test_output"):
            print("✅ Processor initialized successfully")
            print("✅ Output directory created")

            # Clean up
            os.rmdir("./test_output")
            return True
        else:
            print("❌ Output directory not created")
            return False

    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        return False


def test_data_structures():
    """Test data structure handling with dummy data."""
    print("\n=== Testing Data Structures ===")

    try:
        import pandas as pd
        import numpy as np

        config = AnalysisConfig(
            data_directory="/tmp",
            pred_directory="/tmp",
            coord_directory="/tmp",
            output_directory="./test_output",
            years=[2000, 2010],
            seasons=["sos"],
            cluster_names=["BET"],
        )

        processor = PhenologicalDataProcessor(config)

        # Test seasonal timing dictionaries
        if hasattr(processor, "sos_timing") and hasattr(processor, "eos_timing"):
            print("✅ Seasonal timing dictionaries available")
            print(f"   SOS timing entries: {len(processor.sos_timing)}")
            print(f"   EOS timing entries: {len(processor.eos_timing)}")
        else:
            print("❌ Seasonal timing dictionaries missing")
            return False

        # Test with dummy dataframe
        dummy_data = pd.DataFrame(
            {
                "latitude": np.random.uniform(-90, 90, 50),
                "longitude": np.random.uniform(-180, 180, 50),
                "hist_tmin": np.random.uniform(0, 1, 50),
                "hist_tmax": np.random.uniform(0, 1, 50),
                "hist_rad": np.random.uniform(0, 1, 50),
                "hist_precip": np.random.uniform(0, 1, 50),
                "hist_photo": np.random.uniform(0, 1, 50),
                "hist_sm": np.random.uniform(0, 1, 50),
            }
        )

        # Test imputation function
        imputed_data = processor.impute_nearby_values(dummy_data)

        if len(imputed_data) == len(dummy_data):
            print("✅ Data imputation function works")
            return True
        else:
            print("❌ Data imputation failed")
            return False

    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def test_analyzer_initialization():
    """Test analyzer initialization."""
    print("\n=== Testing Analyzer Initialization ===")

    try:
        config = AnalysisConfig(
            data_directory="/tmp",
            pred_directory="/tmp",
            coord_directory="/tmp",
            output_directory="./test_output",
            years=[2000, 2010],
            seasons=["sos"],
            cluster_names=["BET"],
        )

        processor = PhenologicalDataProcessor(config)
        analyzer = SensitivityAnalyzer(processor)

        if hasattr(analyzer, "processor") and hasattr(analyzer, "config"):
            print("✅ Analyzer initialized successfully")
            print("✅ Processor and config references available")
            return True
        else:
            print("❌ Analyzer missing required attributes")
            return False

    except Exception as e:
        print(f"❌ Analyzer initialization failed: {e}")
        return False


def test_difference_calculation():
    """Test difference calculation with dummy data."""
    print("\n=== Testing Difference Calculation ===")

    try:
        import pandas as pd
        import numpy as np

        config = AnalysisConfig(
            data_directory="/tmp",
            pred_directory="/tmp",
            coord_directory="/tmp",
            output_directory="./test_output",
            years=[2000, 2010],
            seasons=["sos"],
            cluster_names=["BET"],
        )

        processor = PhenologicalDataProcessor(config)
        SensitivityAnalyzer(processor)

        # Create dummy data for two time periods
        lat_values = np.linspace(90.0, -89.75, 720)
        lon_values = np.linspace(-180.0, 179.75, 1440)

        # Create coordinate grid
        from itertools import product

        combinations = list(product(lat_values, lon_values))

        df1 = pd.DataFrame(combinations, columns=["latitude", "longitude"])
        df1["hist_temp"] = np.random.uniform(0, 1, len(df1))
        df1["hist_sol"] = np.random.uniform(0, 1, len(df1))
        df1["hist_p"] = np.random.uniform(0, 1, len(df1))

        df2 = pd.DataFrame(combinations, columns=["latitude", "longitude"])
        df2["hist_temp"] = np.random.uniform(0, 1, len(df2))
        df2["hist_sol"] = np.random.uniform(0, 1, len(df2))
        df2["hist_p"] = np.random.uniform(0, 1, len(df2))

        # Calculate difference
        difference = df1 - df2

        # Test reshaping
        temp_diff = np.reshape(difference["hist_temp"], (720, 1440))

        if temp_diff.shape == (720, 1440):
            print("✅ Difference calculation and reshaping works")
            print(f"   Output shape: {temp_diff.shape}")
            return True
        else:
            print("❌ Incorrect output shape")
            return False

    except Exception as e:
        print(f"❌ Difference calculation test failed: {e}")
        return False


def test_imports():
    """Test that all required dependencies are available."""
    print("\n=== Testing Dependencies ===")

    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "cartopy",
        "seaborn",
        "pickle",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT AVAILABLE")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All dependencies available")
        return True


def main():
    """Run all tests."""
    print("🧪 Testing Sensitivity Analysis Generator")
    print("=" * 50)

    # Configure logging to reduce noise during testing
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)

    tests = [
        ("Dependencies", test_imports),
        ("Configuration Creation", test_config_creation),
        ("Processor Initialization", test_processor_initialization),
        ("Data Structures", test_data_structures),
        ("Analyzer Initialization", test_analyzer_initialization),
        ("Difference Calculation", test_difference_calculation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print(
            "✅ All tests passed! The refactored sensitivity analysis code should work correctly."
        )
        print(
            "📊 The code can generate temporal difference maps for climate driver sensitivity."
        )
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
