#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=sos_eos_indices_BET_%j.out
#SBATCH --error=sos_eos_indices_BET_%j.err
#SBATCH --job-name=sos_eos_indices_BET
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0-1:00
#SBATCH --partition=short

# Generate SOS/EOS phenology indices for BET PFT
# Usage: sbatch bashscripts/drivers/generate_sos_eos_indices_BET.sh

module purge
module load anaconda

PFT="BET"

DATA_PATH="/burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/BET_analysisdf_20251106"
DATA_PKL_PATH="/burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/BET_1982_2021.pkl"
CONFIG_PATH="/burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/sos_eos_BET_merged_smoothed.json"
OUTPUT_PATH="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/BET"

echo "========================================================================"
echo "SOS/EOS INDEX GENERATION - ${PFT}"
echo "========================================================================"
echo "Analysis data:  ${DATA_PATH}"
echo "Data pickle:    ${DATA_PKL_PATH}"
echo "Config path:    ${CONFIG_PATH}"
echo "Output path:    ${OUTPUT_PATH}"
echo "========================================================================"
echo ""

python /burg-archive/home/al4385/phenofusion/debugging/save_sos_eos_indices_BET.py

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
