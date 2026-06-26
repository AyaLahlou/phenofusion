#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=attention_window_NDT_%j.out
#SBATCH --error=attention_window_NDT_%j.err
#SBATCH --job-name=attention_window_NDT
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-3:00
#SBATCH -C mem768
#SBATCH --partition=short

# Extract attention window data for NDT PFT
# Usage: sbatch bashscripts/drivers/get_attention_window_NDT.sh
# NOTE: SOS/EOS index files exist at the paths below but are currently empty
#       (shape=(0,)). Regenerate by running debugging/save_sos_eos_indices.py for NDT.

module purge
module load anaconda

PFT="NDT"

DATA_PATH="/burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/NDT_1982_2021.pkl"
PRED_PATH="/burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/NDT_1982_2021.pkl"
COORDS_PATH="/burg-archive/glab/users/al4385/data/coordinates/NDT.parquet"
SOS_INDICES="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/SOS_pred_indices_NDT.npy"
EOS_INDICES="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/EOS_pred_indices_NDT.npy"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/attention_window"
OUTPUT_PREFIX="NDT"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "ATTENTION WINDOW EXTRACTION - ${PFT}"
echo "========================================================================"
echo "Data path:     ${DATA_PATH}"
echo "Pred path:     ${PRED_PATH}"
echo "Coords path:   ${COORDS_PATH}"
echo "SOS indices:   ${SOS_INDICES}"
echo "EOS indices:   ${EOS_INDICES}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "========================================================================"
echo ""

echo "Checking input files..."
for file in "${DATA_PATH}" "${PRED_PATH}" "${COORDS_PATH}" "${SOS_INDICES}" "${EOS_INDICES}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
    echo "  Found: $(basename $file)"
done
echo ""

echo "Starting attention window extraction..."
python /burg-archive/home/al4385/phenofusion/scripts/get_attention_window_data.py \
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
    echo "ATTENTION WINDOW EXTRACTION COMPLETE"
    echo "========================================================================"
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}/${OUTPUT_PREFIX}"*.csv 2>/dev/null || echo "  (no CSV files found)"
else
    echo ""
    echo "========================================================================"
    echo "ATTENTION WINDOW EXTRACTION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
