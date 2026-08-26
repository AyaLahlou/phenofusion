#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=attention_mean_NDT_%j.out
#SBATCH --error=attention_mean_NDT_%j.err
#SBATCH --job-name=attention_mean_NDT
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-3:00

#SBATCH --partition=short

# Extract attention-weighted mean day data for NDT PFT
# Usage: sbatch bashscripts/drivers/get_attention_mean_NDT.sh

module purge
module load anaconda

PFT="NDT"

DATA_PATH="/burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/NDT_1982_2021.pkl"
PRED_PATH="/burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/NDT_1982_2021.pkl"
COORDS_PATH="/burg-archive/glab/users/al4385/data/coordinates/NDT.parquet"
SOS_INDICES="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/SOS_pred_indices_NDT.npy"
EOS_INDICES="/burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/EOS_pred_indices_NDT.npy"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/attention_mean"
OUTPUT_PREFIX="NDT"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "ATTENTION MEAN DAY EXTRACTION - ${PFT}"
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

echo "Starting attention mean day extraction..."
python /burg-archive/home/al4385/phenofusion/scripts/get_attention_mean_data.py \
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
    echo "ATTENTION MEAN DAY EXTRACTION COMPLETE"
    echo "========================================================================"
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}/${OUTPUT_PREFIX}"*.csv 2>/dev/null || echo "  (no CSV files found)"
else
    echo ""
    echo "========================================================================"
    echo "ATTENTION MEAN DAY EXTRACTION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
