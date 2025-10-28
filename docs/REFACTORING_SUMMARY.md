# TFT Metrics Analysis Refactoring

This document summarizes the refactoring changes made to the TFT metrics analysis code to improve maintainability, readability, and robustness.

## Files Modified

### 1. `src/phenofusion/analysis/metrics_tft.py`
**Major refactoring** - Transformed from a monolithic script into a well-structured, object-oriented module.

### 2. `bashscripts/metrics_SHR.sh`
**Complete rewrite** - Enhanced with robust error handling, logging, and best practices.

## Key Improvements

### Python Code (`metrics_tft.py`)

#### 1. **Modular Design**
- **Before**: Single monolithic file with large functions doing multiple things
- **After**: Separated into focused classes with single responsibilities:
  - `MetricsCalculator`: Handles metric calculations
  - `DataProcessor`: Manages data loading and processing
  - `PlotConfiguration`: Manages plot styling and configuration
  - `MetricsPlotter`: Handles plot creation

#### 2. **Error Handling & Validation**
```python
# Before: No error handling
with open(file_path, "rb") as file:
    dictionary = pickle.load(file)

# After: Comprehensive error handling
def load_pickle_file(file_path: Path) -> Dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading pickle file {file_path}: {e}")
```

#### 3. **Type Hints & Documentation**
- **Before**: No type hints, minimal documentation
- **After**: Comprehensive type hints and docstrings for all functions and classes

#### 4. **Configuration Management**
- **Before**: Hardcoded values scattered throughout code
- **After**: Centralized configuration in `PlotConfiguration` class and separate `config.py`

#### 5. **Code Deduplication**
- **Before**: Repetitive plotting code for each subplot
- **After**: Generic plotting methods that handle all subplot types

#### 6. **Improved Data Processing**
```python
# Before: Complex, hard-to-follow data transformations
across_site = df.copy()
across_site = df.groupby(["location_id"]).mean().reset_index()
# ... many more lines

# After: Clear, focused methods
def calculate_site_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    site_agg = df.groupby("location_id").agg({
        'observed': 'mean',
        'pred_05': 'mean'
    }).reset_index()
    site_agg.columns = ["location_id", "observed_site", "predicted_site"]
    return site_agg.sort_values(by="location_id")
```

#### 7. **Better Variable Naming**
- **Before**: `df["Flattened_Values"]`, `accross_site`, `F_MSC`
- **After**: `df["observed"]`, `site_agg`, `observed_seasonal`

#### 8. **Logging Instead of Print Statements**
- **Before**: `print("predictions", file_path)`
- **After**: `logging.info(f"Loading predictions from: {prediction_path}")`

### Bash Script (`metrics_SHR.sh`)

#### 1. **Robust Error Handling**
```bash
# Before: No error handling
module load anaconda
pip install tft-torch

# After: Comprehensive error handling
load_modules() {
    log_info "Loading required modules..."
    module load anaconda || {
        log_error "Failed to load anaconda module"
        return 1
    }
    log_success "Modules loaded successfully"
}
```

#### 2. **Structured Logging**
- Added colored logging functions (`log_info`, `log_error`, `log_warning`, `log_success`)
- Timestamped log messages
- Better job information display

#### 3. **Input Validation**
```bash
check_file_exists() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        log_error "$file_description not found: $file_path"
        return 1
    fi
}
```

#### 4. **Modular Functions**
- Split script into focused functions
- Each function has a single responsibility
- Better code reusability

#### 5. **Enhanced SLURM Integration**
- Added error output file (`#SBATCH --error=...`)
- Job ID in output filenames for uniqueness
- Better job information display

#### 6. **Safety Features**
```bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures
trap 'log_error "Script interrupted"; exit 130' INT TERM
```

## New Files Created

### 1. `src/phenofusion/analysis/config.py`
Centralized configuration management with dataclasses:
- `PlotConfig`: Plot styling and parameters
- `DataConfig`: Data processing configuration
- `MetricsConfig`: Metrics calculation settings

### 2. `tests/test_metrics_tft_refactored.py`
Comprehensive test suite covering:
- Unit tests for each class
- Integration tests for the complete pipeline
- Mock data generation for testing
- Edge case handling

## Benefits of Refactoring

### 1. **Maintainability**
- Modular design makes it easy to modify individual components
- Clear separation of concerns
- Well-documented code with type hints

### 2. **Reliability**
- Comprehensive error handling prevents silent failures
- Input validation ensures data integrity
- Proper logging for debugging

### 3. **Reusability**
- Modular classes can be used independently
- Configuration management allows easy customization
- Generic plotting methods work for different data types

### 4. **Testability**
- Each component can be tested independently
- Mock data capabilities for testing
- Clear interfaces make testing straightforward

### 5. **Performance**
- More efficient data processing
- Better memory management
- Configurable sampling rates for large datasets

## Migration Guide

### For Users of the Original Code

1. **Basic Usage Remains the Same**:
   ```bash
   python metrics_tft.py --filename "data.pkl" --data_dir /path/to/data --pred_dir /path/to/pred --fig_dir /path/to/output
   ```

2. **New Features Available**:
   - Better error messages and logging
   - Configurable plot parameters
   - More robust file handling

3. **For Advanced Users**:
   ```python
   # Can now import and use classes individually
   from metrics_tft import DataProcessor, MetricsPlotter

   df, site_agg = DataProcessor.process_data(pred_path, data_path)
   plotter = MetricsPlotter()
   metrics = plotter.create_metrics_plot(df, site_agg, "Title", output_path)
   ```

### Backward Compatibility

The refactored code maintains full backward compatibility:
- Same command-line interface
- Same output format
- Same plot appearance (unless explicitly configured otherwise)

## Best Practices Implemented

1. **PEP 8 Compliance**: Proper Python style guidelines
2. **Type Safety**: Comprehensive type hints
3. **Error Handling**: Graceful failure with informative messages
4. **Documentation**: Comprehensive docstrings and comments
5. **Testing**: Unit and integration tests
6. **Configuration**: Externalized configuration management
7. **Logging**: Structured logging instead of print statements
8. **Security**: Input validation and safe file operations

## Future Enhancements

The refactored code provides a solid foundation for future improvements:

1. **Configuration Files**: YAML/JSON configuration files
2. **Multiple Output Formats**: PDF, SVG, interactive plots
3. **Advanced Metrics**: Additional statistical measures
4. **Parallel Processing**: Multi-threaded data processing
5. **API Integration**: REST API for remote analysis
6. **Database Support**: Direct database connectivity

## Testing

Run the test suite to verify the refactored code:

```bash
cd /burg-archive/home/al4385/phenofusion
python -m pytest tests/test_metrics_tft_refactored.py -v
```

## Conclusion

This refactoring significantly improves the codebase while maintaining full backward compatibility. The new structure is more maintainable, reliable, and extensible, providing a solid foundation for future development.
