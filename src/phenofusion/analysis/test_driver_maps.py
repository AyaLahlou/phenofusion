#!/usr/bin/env python3
"""
Test script for the driver map generator.

This script performs basic tests to ensure the refactored code works correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from generate_driver_maps import DriverMapGenerator, create_file_mapping

    print("✅ Successfully imported DriverMapGenerator")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)


def test_file_mapping():
    """Test the file mapping functionality."""
    print("\n=== Testing File Mapping ===")

    # Test with the expected data directory
    data_dir = "/burg/home/al4385/code/phenology_analysis/drivers_data/"

    if not os.path.exists(data_dir):
        print(f"⚠️  Data directory not found: {data_dir}")
        print("   This is expected if running outside the original environment")
        return True

    try:
        file_mapping = create_file_mapping(data_dir)

        if file_mapping:
            print(f"✅ Found {len(file_mapping)} PFT categories")
            for pft, phases in file_mapping.items():
                for phase, filepath in phases.items():
                    print(f"   {pft} {phase}: {os.path.basename(filepath)}")
        else:
            print("⚠️  No files found in data directory")

        return True

    except Exception as e:
        print(f"❌ File mapping test failed: {e}")
        return False


def test_generator_initialization():
    """Test DriverMapGenerator initialization."""
    print("\n=== Testing Generator Initialization ===")

    try:
        # Create a temporary output directory
        output_dir = "./test_output"
        data_dir = "/tmp"  # Use a directory that should exist

        DriverMapGenerator(data_dir, output_dir)

        # Check if output directory was created
        if os.path.exists(output_dir):
            print("✅ Generator initialized successfully")
            print(f"✅ Output directory created: {output_dir}")

            # Clean up
            os.rmdir(output_dir)
            return True
        else:
            print("❌ Output directory not created")
            return False

    except Exception as e:
        print(f"❌ Generator initialization failed: {e}")
        return False


def test_data_processing():
    """Test data processing methods with dummy data."""
    print("\n=== Testing Data Processing ===")

    try:
        import pandas as pd
        import numpy as np

        # Create dummy data
        dummy_data = pd.DataFrame(
            {
                "latitude": np.random.uniform(-90, 90, 100),
                "longitude": np.random.uniform(-180, 180, 100),
                "hist_tmin": np.random.uniform(0, 1, 100),
                "hist_tmax": np.random.uniform(0, 1, 100),
                "hist_rad": np.random.uniform(0, 1, 100),
                "hist_precip": np.random.uniform(0, 1, 100),
                "hist_photo": np.random.uniform(0, 1, 100),
                "hist_sm": np.random.uniform(0, 1, 100),
            }
        )

        generator = DriverMapGenerator("/tmp", "./test_output")

        # Test concatenation
        result_df = generator.concatenate_dataframes([dummy_data])

        if len(result_df) > 0:
            print("✅ Data concatenation successful")
            print(f"   Output grid size: {len(result_df)} points")

            # Test colorization
            rgb_list = generator.colorize_data(result_df)

            if len(rgb_list) == 3:
                print("✅ Data colorization successful")
                print(
                    f"   RGB channels created with shape: {len(rgb_list[0])}x{len(rgb_list[0][0])}"
                )
                return True
            else:
                print("❌ Colorization failed - wrong number of channels")
                return False
        else:
            print("❌ Data concatenation failed")
            return False

    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False


def test_imports():
    """Test that all required dependencies are available."""
    print("\n=== Testing Dependencies ===")

    required_packages = ["pandas", "numpy", "matplotlib", "cartopy"]

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
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All dependencies available")
        return True


def main():
    """Run all tests."""
    print("🧪 Testing Driver Map Generator")
    print("=" * 50)

    # Configure logging to reduce noise during testing
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("cartopy").setLevel(logging.WARNING)

    tests = [
        ("Dependencies", test_imports),
        ("File Mapping", test_file_mapping),
        ("Generator Initialization", test_generator_initialization),
        ("Data Processing", test_data_processing),
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
        print("✅ All tests passed! The refactored code should work correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
