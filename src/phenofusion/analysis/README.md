# Phenofusion Analysis Module

This directory contains analysis tools for phenological model interpretation and visualization.

## Overview

This directory contains refactored and cleaned versions of phenological analysis code, providing three main analytical capabilities:

### 1. Driver Map Generation
Creates RGB visualization maps showing the relative importance of three climate drivers:
- **Temperature** (Red channel): Combined min/max temperature
- **Solar Radiation** (Green channel): Combined radiation and photoperiod
- **Precipitation** (Blue channel): Combined precipitation and soil moisture

### 2. Temporal Sensitivity Analysis
Generates temporal difference maps showing how climate driver sensitivity has changed between different years, helping to understand:
- Shifts in temperature sensitivity over time
- Changes in solar radiation importance
- Evolution of precipitation/water availability influence
- Regional patterns of sensitivity change

### 3. Heatmap Analysis 🆕
Creates publication-ready heatmaps combining attention scores with time series data for climate driver analysis:
- **Feature importance heatmaps** with time series overlays showing driver influence over time
- **Attention pattern visualization** revealing temporal focus of deep learning models
- **City-specific analysis** with automatic geographic filtering and seasonal analysis
- **Multi-year comparison** capabilities for understanding temporal dynamics

## Contents

### 📊 Driver Map Analysis
- **`generate_driver_maps.py`** - Main implementation for RGB driver map generation
- **`run_driver_maps.py`** - Simple runner script for driver map analysis
- **`test_driver_maps.py`** - Comprehensive test suite

**Purpose**: Generate RGB visualizations showing the relative importance of climate drivers (temperature, precipitation, solar radiation) in phenological predictions.

### 📈 Sensitivity Analysis
- **`generate_sensitivity_plots.py`** - Temporal sensitivity analysis framework
- **`run_sensitivity_analysis.py`** - Simple runner for sensitivity analysis
- **`test_sensitivity_analysis.py`** - Test suite for sensitivity analysis

**Purpose**: Compare climate driver importance between different time periods to understand temporal dynamics.

### 🔥 Heatmap Analysis
- **`heatmap_analysis.py`** - Feature importance and attention heatmap generation
- **`run_heatmap_analysis.py`** - Command-line interface for heatmap analysis
- **`test_heatmap_analysis.py`** - Comprehensive test suite

**Purpose**: Create publication-ready heatmaps combining attention scores with time series data for climate driver analysis.

### 📝 Legacy/Original Files
- **`drivermap_july2025.ipynb`** - Original notebook (refactored into driver maps)
- **`sensitivity_plot.py`** - Original sensitivity script (refactored)
- **`metrics_tft.py`** - TFT model metrics and evaluation
- **`heatmap.ipynb`** - Original heatmap notebook (refactored into heatmap_analysis.py)

## Quick Start

### Driver Map Analysis
```bash
# Basic usage
python run_driver_maps.py --attention-file /path/to/attention.pkl --output-dir ./plots

# With custom settings
python run_driver_maps.py \
    --attention-file /path/to/attention.pkl \
    --coordinate-file /path/to/coordinates.parquet \
    --year 2020 \
    --pft "BDT" \
    --output-dir ./custom_plots
```

### Sensitivity Analysis
```bash
# Compare two years
python run_sensitivity_analysis.py \
    --attention-file-1 /path/to/2020_attention.pkl \
    --attention-file-2 /path/to/2060_attention.pkl \
    --year-1 2020 \
    --year-2 2060

# With custom coordinate file
python run_sensitivity_analysis.py \
    --attention-file-1 /path/to/file1.pkl \
    --attention-file-2 /path/to/file2.pkl \
    --coordinate-file /path/to/coords.parquet \
    --output-dir ./sensitivity_plots
```

### Heatmap Analysis
```bash
# Basic city analysis
python run_heatmap_analysis.py --city "Budapest" --pft "BDT" --season "sos"

# Advanced options
python run_heatmap_analysis.py \
    --city "Tokyo" \
    --pft "BET" \
    --season "eos" \
    --year 2006 \
    --output-dir "./plots" \
    --verbose
```

## Python API Usage

### Driver Maps
```python
from phenofusion.analysis.generate_driver_maps import DriverMapGenerator

generator = DriverMapGenerator()
generator.generate_rgb_driver_map(
    attention_file="/path/to/attention.pkl",
    coordinate_file="/path/to/coordinates.parquet",
    output_file="driver_map.png"
)
```

### Sensitivity Analysis
```python
from phenofusion.analysis.generate_sensitivity_plots import PhenologicalDataProcessor

processor = PhenologicalDataProcessor()
processor.create_temporal_sensitivity_comparison(
    attention_file_1="/path/to/2020.pkl",
    attention_file_2="/path/to/2060.pkl",
    coordinate_file="/path/to/coordinates.parquet",
    output_file="sensitivity_comparison.png"
)
```

### Heatmap Analysis
```python
from phenofusion.analysis.heatmap_analysis import analyze_city_heatmaps

# Complete city analysis
analyze_city_heatmaps("Budapest", pft="BDT", season="sos")

# Custom analysis
from phenofusion.analysis.heatmap_analysis import HeatmapAnalyzer
analyzer = HeatmapAnalyzer(output_dir="./custom_plots")
# ... use analyzer methods
```

## Files

### Driver Map Generation
- `generate_driver_maps.py`: Main class-based implementation for RGB driver maps
- `run_driver_maps.py`: Simple runner script for driver maps
- `test_driver_maps.py`: Test suite for driver map functionality

### Sensitivity Analysis
- `generate_sensitivity_plots.py`: Main implementation for temporal sensitivity analysis
- `run_sensitivity_analysis.py`: Simple runner script for sensitivity analysis
- `test_sensitivity_analysis.py`: Test suite for sensitivity analysis

### Documentation
- `README.md`: This comprehensive documentation file

## Features

### Clean Architecture
- Object-oriented design with clear separation of concerns
- Comprehensive error handling and logging
- Type hints for better code maintainability
- Configurable parameters

### Data Processing
- Automatic file discovery and mapping
- Latitude-based filtering for different Plant Functional Types (PFTs)
- Data concatenation and normalization
- Grid interpolation for global coverage

### Visualization
- Multiple map projections (PlateCarree, Robinson)
- Cartographic features (coastlines, gridlines)
- Customizable output resolution and cropping
- Robust handling of projection artifacts

### Supported PFTs
- **BET**: Broadleaf Evergreen Trees
- **NET**: Needleleaf Evergreen Trees
- **NDT**: Needleleaf Deciduous Trees
- **BDT**: Broadleaf Deciduous Trees (multiple latitude bands)

## Usage

### Driver Map Generation

#### Quick Start (Recommended)

The simplest way to generate all available driver maps:

```bash
# Navigate to the analysis directory
cd /burg-archive/home/al4385/phenofusion/src/phenofusion/analysis/

# Run with default settings
python run_driver_maps.py

# Or specify custom directories
python run_driver_maps.py \
    --data-dir /path/to/your/csv/files/ \
    --output-dir ./my_maps/ \
    --projection Robinson
```

#### Advanced Usage

For more control over driver map generation:

```bash
python generate_driver_maps.py \
    --data-dir /burg/home/al4385/code/phenology_analysis/drivers_data/ \
    --output-dir ./driver_maps_output/ \
    --projection PlateCarree \
    --dpi 300 \
    --show-plots
```

### Temporal Sensitivity Analysis

#### Quick Start

Generate temporal difference maps comparing two years:

```bash
# Run with default settings (1985 vs 2020)
python run_sensitivity_analysis.py

# Or specify custom years and seasons
python run_sensitivity_analysis.py \
    --year1 1985 \
    --year2 2020 \
    --seasons sos eos \
    --output-dir ./sensitivity_output/
```

#### Advanced Usage

For more control over sensitivity analysis:

```bash
python generate_sensitivity_plots.py \
    --data-dir /burg/glab/users/al4385/data/TFT_30_40years/ \
    --pred-dir /burg/glab/users/al4385/predictions/TFT_30_40years/ \
    --coord-dir /burg/glab/users/al4385/data/coordinates/ \
    --year1 1985 \
    --year2 2020 \
    --seasons sos eos \
    --show-plots
```

### Python API

You can also use the classes directly in your own scripts:

#### Driver Maps
```python
from generate_driver_maps import DriverMapGenerator

# Initialize
generator = DriverMapGenerator(
    data_dir="/path/to/csv/files/",
    output_dir="./output/"
)

# Generate a specific map
generator.generate_pft_map(
    file_paths=["BET_SOS_data.csv"],
    pft_names=["BET"],
    output_name="BET_SOS_Map",
    projection="Robinson",
    dpi=300
)
```

#### Sensitivity Analysis
```python
from generate_sensitivity_plots import (
    AnalysisConfig, PhenologicalDataProcessor, SensitivityAnalyzer
)

# Create configuration
config = AnalysisConfig(
    data_directory="/path/to/data/",
    pred_directory="/path/to/predictions/",
    coord_directory="/path/to/coordinates/",
    output_directory="./output/",
    years=[1985, 2020],
    seasons=["sos", "eos"],
    cluster_names=["BET", "NET", "NDT"]
)

# Initialize components
processor = PhenologicalDataProcessor(config)
analyzer = SensitivityAnalyzer(processor)

# Calculate temporal differences
temp_diff, solar_diff, precip_diff = analyzer.calculate_temporal_differences(
    1985, 2020, "sos"
)
```

## Expected Input Data

### Driver Map Generation

The script expects CSV files with the following columns:
- `latitude`: Latitude coordinates
- `longitude`: Longitude coordinates
- `hist_tmin`: Historical minimum temperature weights
- `hist_tmax`: Historical maximum temperature weights
- `hist_rad`: Historical radiation weights
- `hist_precip`: Historical precipitation weights
- `hist_photo`: Historical photoperiod weights
- `hist_sm`: Historical soil moisture weights

#### File Naming Convention

The script automatically detects files using these patterns:
- `BET_*_SOS*.csv` / `BET_*_EOS*.csv`
- `NET_*_SOS*.csv` / `NET_*_EOS*.csv`
- `NDT_*_SOS*.csv` / `NDT_*_EOS*.csv`
- `BDT_50_20_*_SOS*.csv` / `BDT_50_20_*_EOS*.csv` (etc.)

### Sensitivity Analysis

The script expects:
- **Data files**: Pickle files containing model training data
- **Prediction files**: Pickle files containing model predictions and attention weights
- **Coordinate files**: Parquet files mapping location IDs to coordinates

#### Directory Structure
```
data_directory/
├── BET.pickle
├── NET.pickle
├── NDT.pickle
└── BDT_*.pickle

pred_directory/
├── pred_BET.pkl
├── pred_NET.pkl
├── pred_NDT.pkl
└── pred_BDT_*.pkl

coord_directory/
├── BET.parquet
├── NET.parquet
├── NDT.parquet
└── BDT_*.parquet
```

## Output

### Driver Map Generation

The script generates several types of maps:

1. **Individual PFT Maps**: Separate maps for each plant functional type and phase
   - `BET_SOS.png`, `BET_EOS.png`
   - `NET_SOS.png`, `NET_EOS.png`
   - `NDT_SOS.png`, `NDT_EOS.png`

2. **Combined BDT Maps**: All broadleaf deciduous tree regions combined
   - `BDT_Combined_SOS.png`
   - `BDT_Combined_EOS.png`

3. **Comprehensive Maps**: All PFTs combined
   - `All_PFTs_SOS.png`
   - `All_PFTs_EOS.png`

### Sensitivity Analysis

The script generates temporal difference maps:

1. **Temperature Sensitivity**: Changes in temperature driver importance
   - `temperature_sensitivity_difference_1985_2020_sos.png`
   - `temperature_sensitivity_difference_1985_2020_eos.png`

2. **Solar Radiation Sensitivity**: Changes in solar driver importance
   - `solar_radiation_sensitivity_difference_1985_2020_sos.png`
   - `solar_radiation_sensitivity_difference_1985_2020_eos.png`

3. **Water Availability Sensitivity**: Changes in precipitation/moisture importance
   - `water_availability_sensitivity_difference_1985_2020_sos.png`
   - `water_availability_sensitivity_difference_1985_2020_eos.png`

## Configuration Options

### Command Line Arguments

- `--data-dir`: Directory containing CSV files
- `--output-dir`: Directory for output plots
- `--show-plots`: Display plots instead of just saving
- `--projection`: Map projection (`PlateCarree` or `Robinson`)
- `--dpi`: Image resolution (default: 300)

### Key Parameters

- **Latitude Filtering**: Automatically applied based on PFT
- **Grid Resolution**: 720x1440 (0.25° resolution)
- **Color Scaling**: RGB values normalized to 0-255 range
- **Crop Latitude**: -60° (hides Antarctica by default)

## Improvements Over Original Notebook

1. **Code Organization**: Clear class structure vs. scattered functions
2. **Error Handling**: Comprehensive try/catch blocks and logging
3. **Documentation**: Extensive docstrings and comments
4. **Flexibility**: Configurable parameters vs. hardcoded values
5. **Robustness**: Handles missing files and invalid data gracefully
6. **Maintainability**: Type hints and clear variable names
7. **Reusability**: Modular design allows easy extension

## Dependencies

Required Python packages:
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `cartopy`: Cartographic projections
- `pathlib`: Path handling

Install with:
```bash
pip install pandas numpy matplotlib cartopy
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Check that CSV files exist in the specified data directory
2. **Permission Errors**: Ensure write permissions for the output directory
3. **Memory Issues**: Large datasets may require more RAM; consider processing subsets
4. **Projection Errors**: Some projections may fail with certain data ranges

### Logging

The script provides detailed logging information. Check the console output for:
- File loading status
- Data filtering results
- Processing progress
- Error messages

### Performance Tips

- Use PlateCarree projection for faster rendering
- Reduce DPI for faster processing during development
- Process individual PFTs separately for debugging

## Example Output

### Driver Maps
The generated RGB driver maps show:
- **Red regions**: Temperature-dominated areas
- **Green regions**: Solar radiation-dominated areas
- **Blue regions**: Precipitation-dominated areas
- **Mixed colors**: Areas with balanced driver influence
- **White areas**: No data available

The RGB visualization provides an intuitive way to understand the relative importance of different climate drivers across global vegetation patterns.

### Sensitivity Analysis
The temporal difference maps show:
- **Red regions**: Increased sensitivity to the driver over time
- **Blue regions**: Decreased sensitivity to the driver over time
- **White regions**: No change in sensitivity
- **Color intensity**: Magnitude of sensitivity change

These maps help identify regions where climate change has altered the relative importance of different environmental drivers for vegetation phenology.

## Testing

Both analysis tools include comprehensive test suites:

```bash
# Test driver map generation
python test_driver_maps.py

# Test sensitivity analysis
python test_sensitivity_analysis.py
```

The test scripts validate:
- Module imports and dependencies
- Data structure handling
- Configuration management
- Core functionality with dummy data

## Comparison with Original Code

### Improvements Over Original Scripts

1. **Code Organization**: Clear class structure vs. scattered functions
2. **Error Handling**: Comprehensive try/catch blocks and logging
3. **Documentation**: Extensive docstrings and user guides
4. **Flexibility**: Configurable parameters vs. hardcoded values
5. **Robustness**: Handles missing files and invalid data gracefully
6. **Maintainability**: Type hints and clear variable names
7. **Reusability**: Modular design allows easy extension
8. **Testing**: Comprehensive test suites for validation

### Original vs. Refactored

| Aspect | Original | Refactored |
|--------|----------|------------|
| Structure | Monolithic scripts | Object-oriented classes |
| Error Handling | Minimal | Comprehensive logging |
| Configuration | Hardcoded values | Configurable parameters |
| Documentation | Basic comments | Full API documentation |
| Testing | None | Complete test suites |
| Reusability | Script-specific | Modular components |
| Maintenance | Difficult | Well-structured |
