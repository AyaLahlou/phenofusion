#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=sos_eos_indices_SHR_%j.out
#SBATCH --error=sos_eos_indices_SHR_%j.err
#SBATCH --job-name=sos_eos_indices_SHR
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-1:00
#SBATCH --partition=short

# Generate SOS/EOS phenology indices for SHR PFT
# Usage: sbatch bashscripts/drivers/generate_sos_eos_indices_SHR.sh

module purge
module load anaconda

PFT="SHR"

SCRIPT_PATH="/burg-archive/home/al4385/phenofusion/debugging/save_sos_eos_indices_SHR.py"
DATA_PATH="/burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/SHR_analysisdf_20251106"
CONFIG_PATH="/burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/sos_eos_SHR_merged_smoothed.json"
DATA_PKL_PATH="/burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/SHR_1982_2021.pkl"
OUTPUT_PATH="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/SHR"

echo "========================================================================"
echo "SOS/EOS INDEX GENERATION - ${PFT}"
echo "========================================================================"
echo "Script:      ${SCRIPT_PATH}"
echo "Data path:   ${DATA_PATH}"
echo "Config path: ${CONFIG_PATH}"
echo "Data pkl:    ${DATA_PKL_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "========================================================================"
echo ""

echo "Starting SOS/EOS index generation..."
python "${SCRIPT_PATH}"

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
