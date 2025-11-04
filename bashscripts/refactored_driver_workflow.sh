#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=driver_workflow
#SBATCH --output=driver_workflow_%j.out
#SBATCH --error=driver_workflow_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-15:00
#SBATCH -C mem768

# Complete workflow for generating driver maps
# This script:
# 1. Extracts driver data from predictions
# 2. Generates RGB driver maps

set -e  # Exit on error

echo "========================================="
echo "Driver Maps Generation Workflow"
echo "Started at $(date)"
echo "========================================="

# Load modules
module load anaconda
module load cuda11.1/toolkit
pip install scipy  # For spatial interpolation

# Configuration
PFT="BET"  # Change this to your PFT
DATA_DIR="/burg/glab/users/al4385/data/TFT_40_overlapping_samples"
PRED_DIR="/burg/glab/users/al4385/predictions/pred_40year_moresamples"
COORD_DIR="/burg/glab/users/al4385/data/coordinates"
OUTPUT_DIR="/burg/glab/users/al4385/analysis/drivers_data_refactored"
MAP_OUTPUT_DIR="/burg/glab/users/al4385/figures/drivermaps_refactored"
FORECAST_WINDOW=30

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MAP_OUTPUT_DIR"

# Step 1: Extract driver data from predictions
echo ""
echo "========================================="
echo "Step 1: Extracting driver data for $PFT"
echo "========================================="

DATA_PATH="$DATA_DIR/${PFT}_1982_2021.pkl"
PRED_PATH="$PRED_DIR/${PFT}/${PFT}_1982_2021.pkl"
COORD_PATH="$COORD_DIR/${PFT}.parquet"
OUTPUT_BASE="$OUTPUT_DIR/${PFT}_$(date '+%Y%m%d')"

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$PRED_PATH" ]; then
    echo "Error: Prediction file not found: $PRED_PATH"
    exit 1
fi

if [ ! -f "$COORD_PATH" ]; then
    echo "Error: Coordinate file not found: $COORD_PATH"
    exit 1
fi

python /burg-archive/home/al4385/phenofusion/src/phenofusion/dataio/driversdata_refactored.py \
    --PFT "$PFT" \
    --pred_path "$PRED_PATH" \
    --data_path "$DATA_PATH" \
    --coord_path "$COORD_PATH" \
    --output_path "$OUTPUT_BASE" \
    --forecast_window_length "$FORECAST_WINDOW"

if [ $? -ne 0 ]; then
    echo "Error: Driver data extraction failed"
    exit 1
fi

echo "Driver data extraction completed successfully"

# Step 2: Generate driver maps
echo ""
echo "========================================="
echo "Step 2: Generating driver maps"
echo "========================================="

python /burg-archive/home/al4385/phenofusion/src/phenofusion/analysis/run_driver_maps_refactored.py \
    --data-dir "$OUTPUT_DIR" \
    --output-dir "$MAP_OUTPUT_DIR" \
    --interpolate \
    --projection PlateCarree \
    --dpi 300 \
    --crop-lat -60

if [ $? -ne 0 ]; then
    echo "Error: Driver map generation failed"
    exit 1
fi

echo "Driver map generation completed successfully"

# Summary
echo ""
echo "========================================="
echo "Workflow completed at $(date)"
echo "========================================="
echo "Driver data CSV files: $OUTPUT_DIR"
echo "Driver maps: $MAP_OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/${PFT}*.csv
echo ""
ls -lh "$MAP_OUTPUT_DIR"/${PFT}*.png

exit 0
