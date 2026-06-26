#!/bin/bash
#
# Diagnostic script to investigate sharp transitions at latitude band boundaries
#
#SBATCH --account=glab
#SBATCH --job-name=diag_lat
#SBATCH --output=diagnose_latitude_%j.out
#SBATCH -c 1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-01:00

set -e

echo "Starting latitude band diagnostic at $(date)"

module load anaconda
pip install seaborn scipy

# Configuration
BAND1="BDT_50_20"
BAND2="BDT_50_90"
DATA_DIR="/burg/glab/users/ms7073/analysis/driversdata/oversampling/"
OUTPUT_DIR="/burg/glab/users/al4385/figures/diagnostics/"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run diagnostic (date is auto-detected from available files)
python /burg-archive/home/$USER/phenofusion/src/phenofusion/analysis/diagnose_latitude_bands.py \
    --band1 "$BAND1" \
    --band2 "$BAND2" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Diagnostic completed at $(date)"
echo "Check output at: $OUTPUT_DIR"
