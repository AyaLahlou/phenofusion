# Heatmap Analysis Module Documentation

## Overview

The Heatmap Analysis module provides comprehensive tools for analyzing temporal feature importance and attention patterns in phenological forecasting models. It generates publication-ready heatmaps that combine attention scores with time series data for climate driver analysis.

## Features

- **Feature Importance Visualization**: Create heatmaps showing how different climate drivers (temperature, precipitation, solar radiation, etc.) influence phenological predictions over time
- **Attention Pattern Analysis**: Visualize temporal attention patterns from deep learning models to understand which time periods are most important for predictions
- **Geographic Analysis**: Filter and analyze data for specific cities or geographic regions
- **Seasonal Analysis**: Focus on start-of-season (SOS) or end-of-season (EOS) phenological events
- **Multi-year Analysis**: Analyze specific years or aggregate across multiple years
- **Publication-ready Output**: High-quality plots suitable for research publications

## Architecture

### Core Classes

#### `HeatmapAnalyzer`
Main class for generating feature importance and attention heatmaps.

**Key Methods:**
- `generate_feature_importance_heatmap()`: Creates combined heatmap/time series visualization
- `generate_attention_heatmap()`: Creates attention pattern visualization
- `extract_feature_importance_data()`: Processes model weights into importance scores
- `extract_timeseries_data()`: Processes and normalizes time series data

#### `SeasonalDateFinder`
Utility class for determining growing season dates based on geographic location.

**Key Methods:**
- `start_of_growing_season()`: Determines SOS months based on latitude
- `end_of_growing_season()`: Determines EOS months based on latitude
- `find_date_range()`: Finds optimal analysis periods with sufficient data

#### `GeographicLocalizer`
Utility class for geographic data filtering and location services.

**Key Methods:**
- `localize_df()`: Filters data by geographic bounds
- `get_city_bounding_box()`: Gets coordinates for city-based analysis
- `check_latitude_interval()`: Categorizes latitude for data file selection

### Key Functions

#### `create_analysis_dataframe()`
Loads and merges prediction data with coordinates for analysis.

#### `analyze_city_heatmaps()`
Complete analysis pipeline for a city, combining all components.

## Usage Examples

### Basic Usage

```python
from phenofusion.analysis.heatmap_analysis import analyze_city_heatmaps

# Analyze start-of-season for Budapest using BDT (Broadleaf Deciduous Trees)
analyze_city_heatmaps("Budapest", pft="BDT", season="sos")

# Analyze end-of-season for Tokyo in a specific year
analyze_city_heatmaps("Tokyo", pft="BDT", season="eos", specific_year=2006)
```

### Advanced Usage

```python
from phenofusion.analysis.heatmap_analysis import (
    HeatmapAnalyzer,
    create_analysis_dataframe,
    GeographicLocalizer
)

# Custom analysis with specific data paths
data_path = "/path/to/data.pickle"
pred_path = "/path/to/predictions.pkl"
coord_path = "/path/to/coordinates.parquet"

# Create analysis DataFrame
df = create_analysis_dataframe(data_path, pred_path, coord_path)

# Filter by geographic region
lat_range, lon_range = GeographicLocalizer.get_city_bounding_box("Berlin")
df_filtered = GeographicLocalizer.localize_df(df, lat_range, lon_range)

# Initialize analyzer with custom output directory
analyzer = HeatmapAnalyzer(output_dir="./custom_plots")

# Generate visualizations
analyzer.generate_feature_importance_heatmap(
    df_filtered, pred_path, data_path, season="sos", title="Berlin"
)
analyzer.generate_attention_heatmap(
    df_filtered, pred_path, season="sos", title="Berlin"
)
```

### Command Line Usage

```bash
# Basic analysis
python run_heatmap_analysis.py --city "Budapest" --pft "BDT" --season "sos"

# Advanced options
python run_heatmap_analysis.py \
    --city "Tokyo" \
    --pft "BET" \
    --season "eos" \
    --year 2006 \
    --output-dir "./plots" \
    --verbose

# Custom data directories
python run_heatmap_analysis.py \
    --city "Moscow" \
    --data-dir "/custom/data/path" \
    --pred-dir "/custom/predictions/path" \
    --coord-dir "/custom/coordinates/path"
```

## Input Data Requirements

### Data File Structure

The module expects three types of input files:

1. **Data File** (`.pickle`): Original processed data dictionary
   ```python
   {
       "data_sets": {
           "test": {
               "id": array,  # Location_timestamp identifiers
               "target": array,  # Ground truth CSIF values
               "historical_ts_numeric": array,  # Historical time series
               "future_ts_numeric": array  # Future time series
           }
       }
   }
   ```

2. **Predictions File** (`.pkl`): Model predictions and attention weights
   ```python
   {
       "predicted_quantiles": array,  # Predicted values
       "historical_selection_weights": list,  # Feature importance weights (historical)
       "future_selection_weights": list,  # Feature importance weights (future)
       "attention_scores": list  # Attention matrices
   }
   ```

3. **Coordinates File** (`.parquet`): Location coordinates
   ```python
   pd.DataFrame({
       "location": int,  # Location ID
       "latitude": float,  # Latitude coordinates
       "longitude": float  # Longitude coordinates
   })
   ```

### Feature Order

The module expects climate drivers in the following order:
1. **CSIF** (Chlorophyll/Solar-Induced Fluorescence)
2. **Tmin** (Minimum Temperature)
3. **Tmax** (Maximum Temperature)
4. **SR** (Solar Radiation)
5. **PR** (Precipitation)
6. **PP** (Photoperiod)
7. **SM** (Soil Moisture)

## Output

### Feature Importance Heatmap

Generates a multi-panel visualization showing:
- **CSIF Panel**: Observed vs predicted CSIF time series
- **Climate Driver Panels**: Feature importance heatmaps overlaid with time series data
- **Temporal Axis**: Time progression with month labels
- **Prediction Boundary**: Vertical line at day 365 separating historical/future data

### Attention Heatmap

Creates a 2D heatmap showing:
- **X-axis**: Time (relative to prediction point)
- **Y-axis**: Forecast horizon
- **Color Intensity**: Attention weight magnitude
- **Prediction Boundary**: Vertical line separating historical/future periods

### File Outputs

- `feature_importance_{city_name}.png`: Feature importance visualization
- `attention_{city_name}.png`: Attention pattern visualization
- High resolution (300 DPI) suitable for publications

## Configuration

### Plant Functional Types (PFT)

Supported vegetation types:
- **BDT**: Broadleaf Deciduous Trees
- **BET**: Broadleaf Evergreen Trees
- **NDT**: Needleleaf Deciduous Trees
- **NET**: Needleleaf Evergreen Trees
- **GRA**: Grasslands

### Seasons

- **SOS**: Start of Season (spring phenology)
- **EOS**: End of Season (autumn phenology)

### Geographic Coverage

The module automatically determines data file selection based on latitude:
- **50_90**: Northern high latitudes (50°N to 90°N)
- **50_20**: Northern mid-latitudes (20°N to 50°N)
- **-20_20**: Equatorial regions (20°S to 20°N)
- **-20_-60**: Southern latitudes (20°S to 60°S)

## Error Handling

The module includes comprehensive error handling for:
- **Missing Files**: Clear error messages for missing data files
- **Invalid Cities**: Geocoding failures with helpful suggestions
- **Insufficient Data**: Warnings when not enough sites are available
- **Invalid Parameters**: Validation of season types, years, and PFT values
- **Data Structure**: Validation of expected data formats

## Performance Considerations

### Memory Usage
- Large datasets are processed in chunks where possible
- Temporary arrays are cleaned up automatically
- Memory usage scales with number of selected sites

### Processing Time
- City analysis typically takes 1-5 minutes depending on data size
- Most time spent on data loading and plot generation
- Multiple cities can be processed in parallel

### Output Size
- PNG files are typically 2-5 MB at 300 DPI
- Output directory size scales with number of analyses

## Dependencies

### Required Packages
```python
pandas >= 1.3.0
numpy >= 1.20.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
pickle  # Standard library
pathlib  # Standard library
datetime  # Standard library
logging  # Standard library
```

### Optional Packages
```python
geopy >= 2.0.0  # For city geocoding
pytest >= 6.0.0  # For running tests
```

## Testing

The module includes comprehensive tests covering:
- Unit tests for all major functions
- Integration tests for complete workflows
- Edge case handling
- Mock data testing
- Error condition testing

Run tests with:
```bash
pytest tests/test_heatmap_analysis.py -v
```

## Troubleshooting

### Common Issues

1. **City Not Found**
   ```
   Error: City 'CityName' not found
   Solution: Check spelling, try alternative names, or use coordinates directly
   ```

2. **No Suitable Date Range**
   ```
   Warning: No suitable date range found
   Solution: Check if data exists for the specified year/season combination
   ```

3. **Missing Data Files**
   ```
   Error: File not found: /path/to/file
   Solution: Verify file paths and ensure all required files exist
   ```

4. **Insufficient Sites**
   ```
   Warning: Not enough sites for analysis
   Solution: Expand geographic bounds or try different time periods
   ```

### Debug Mode

Enable verbose logging for detailed troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use command line flag:
```bash
python run_heatmap_analysis.py --city "Berlin" --verbose
```

## Contributing

To contribute to the Heatmap Analysis module:

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Add docstrings for all functions and classes
3. **Testing**: Include tests for new functionality
4. **Logging**: Use the logging module for debug information
5. **Error Handling**: Include appropriate error handling and validation

## Version History

- **v1.0**: Initial implementation with basic heatmap generation
- **v1.1**: Added attention pattern visualization
- **v1.2**: Enhanced geographic filtering and city support
- **v1.3**: Added command-line interface and comprehensive testing
- **v1.4**: Improved error handling and documentation

## License

This module is part of the Phenofusion package and follows the same licensing terms.
