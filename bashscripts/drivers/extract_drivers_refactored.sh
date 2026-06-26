#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=extract_drivers_%j.out
#SBATCH --error=extract_drivers_%j.err
#SBATCH --job-name=extract_drivers
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-3:00
#SBATCH -C mem768
#SBATCH --partition=short


# Extract climate driver importance from TFT predictions
# Usage: sbatch bashscripts/extract_drivers_refactored.sh

# Load required modules
module purge
module load anaconda

# Configuration - EDIT THESE FOR YOUR RUN
PFT="BDT"


DATA_PATH="/burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl"
PRED_PATH="/burg/glab/users/al4385/predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl"
COORDS_PATH="/burg/glab/users/al4385/data/coordinates/merged_BDT.parquet"
INDICES_DIR="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_subset_data"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/drivers_data"



SOS_INDICES="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_subset_data/SOS_pred_indices_BDT_1.npy"
EOS_INDICES="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_subset_data/EOS_pred_indices_BDT_1.npy"

# Output settings
OUTPUT_PREFIX="${PFT}_drivers_3"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Print configuration
echo "========================================================================"
echo "DRIVER EXTRACTION - ${PFT}"
echo "========================================================================"
echo "Data path:    ${DATA_PATH}"
echo "Pred path:    ${PRED_PATH}"
echo "Coords path:  ${COORDS_PATH}"
echo "SOS indices:  ${SOS_INDICES}"
echo "EOS indices:  ${EOS_INDICES}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "========================================================================"
echo ""

# Check if input files exist
echo "Checking input files..."
for file in "${DATA_PATH}" "${PRED_PATH}" "${SOS_INDICES}" "${EOS_INDICES}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
    echo "  ✓ Found: $(basename $file)"
done

# Check coords file (may be .csv or .parquet)
if [ ! -f "${COORDS_PATH}" ]; then
    # Try .parquet extension
    COORDS_PATH="${COORDS_DIR}/${PFT}_coords.parquet"
    if [ ! -f "${COORDS_PATH}" ]; then
        echo "ERROR: Coordinates file not found (tried .csv and .parquet)"
        exit 1
    fi
fi
echo "  ✓ Found: $(basename ${COORDS_PATH})"
echo ""

# Run extraction
echo "Starting driver extraction..."
python /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/driver_data_modified.py \
    --data-path "${DATA_PATH}" \
    --pred-path "${PRED_PATH}" \
    --coords-path "${COORDS_PATH}" \
    --sos-indices "${SOS_INDICES}" \
    --eos-indices "${EOS_INDICES}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --pft "${PFT}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ DRIVER EXTRACTION COMPLETE"
    echo "========================================================================"
    echo "Output files:"
    ls -lh ${OUTPUT_DIR}/${OUTPUT_PREFIX}_*.csv
    echo ""
    echo "Diagnostic plots:"
    ls -lh ${OUTPUT_DIR}/${PFT}_*.png
else
    echo ""
    echo "========================================================================"
    echo "✗ DRIVER EXTRACTION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
