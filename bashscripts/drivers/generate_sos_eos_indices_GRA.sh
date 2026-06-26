#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=sos_eos_indices_GRA_%j.out
#SBATCH --error=sos_eos_indices_GRA_%j.err
#SBATCH --job-name=sos_eos_indices_GRA
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0-1:00
#SBATCH --partition=short

# Generate SOS/EOS phenology indices for GRA PFT
# Usage: sbatch bashscripts/drivers/generate_sos_eos_indices_GRA.sh

module purge
module load anaconda

PFT="GRA"

DATA_PATH="/burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/GRA_analysisdf_20251106"
CONFIG_PATH="/burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/sos_eos_GRA_merged_smoothed.json"
DATA_PKL_PATH="/burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/GRA_1982_2021.pkl"
OUTPUT_PATH="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/GRA"
FIGURE_OUTPUT_PATH="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_figures/sos_eos_indices/GRA"

echo "========================================================================"
echo "SOS/EOS INDEX GENERATION - ${PFT}"
echo "========================================================================"
echo "Data path:         ${DATA_PATH}"
echo "Config path:       ${CONFIG_PATH}"
echo "Data pkl path:     ${DATA_PKL_PATH}"
echo "Output path:       ${OUTPUT_PATH}"
echo "Figure output:     ${FIGURE_OUTPUT_PATH}"
echo "========================================================================"
echo ""

echo "Starting SOS/EOS index generation..."
python /burg-archive/home/al4385/phenofusion/debugging/save_sos_eos_indices_GRA.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "SOS/EOS INDEX GENERATION COMPLETE"
    echo "========================================================================"
    echo "Output files:"
    ls -lh "${OUTPUT_PATH}"/*.npy 2>/dev/null || echo "  (no .npy files found)"
else
    echo ""
    echo "========================================================================"
    echo "SOS/EOS INDEX GENERATION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
