#!/bin/bash

#SBATCH --account=glab
#SBATCH --job-name=metrGRA
#SBATCH --output=GRA_metrics_%j.out
#SBATCH --error=GRA_metrics_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=68G
#SBATCH --time=0-15:00

set -e  # Exit on error

echo "Starting TFT Metrics Analysis for GRA at $(date)"

# Load modules
module load anaconda
module load cuda11.1/toolkit

# Install packages
pip install --upgrade pip
pip install tft-torch mpl-scatter-density

# Configuration
PFT="GRA"
FILENAME="GRA_20251013.pkl"
DATA_DIR="/burg/glab/users/al4385/data/TFT_30/sorted_GRA.pkl"
PRED_DIR="/burg/glab/users/al4385/predictions/pretrained_1219/GRA_20251013.pkl"
METRICS_OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/metrics_output/"

# Create output directory and navigate to script directory
mkdir -p "$METRICS_OUTPUT_DIR"
cd "/burg-archive/home/al4385/phenofusion/src/phenofusion/analysis"

# Run analysis
python metrics_tft.py \
    --filename "$FILENAME" \
    --data_dir "$DATA_DIR" \
    --pred_dir "$PRED_DIR" \
    --fig_dir "${METRICS_OUTPUT_DIR}metrics_${PFT}.png"

echo "Analysis completed at $(date)"
