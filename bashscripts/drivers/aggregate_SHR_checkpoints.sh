#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=aggregate_SHR_%j.out
#SBATCH --error=aggregate_SHR_%j.err
#SBATCH --job-name=aggregate_SHR
#SBATCH --mem-per-cpu=96G
#SBATCH --time=0-4:00
#SBATCH --partition=short

# Rebuild SHR_1982_2021.pkl from checkpoints (existing merged file is truncated).
# Usage: sbatch bashscripts/drivers/aggregate_SHR_checkpoints.sh

module purge
module load anaconda

CHECKPOINT_FOLDER="/burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/SHR"
PRED_FILENAME="SHR_20251023.pkl"
OUT_PATH="/burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/SHR_1982_2021.pkl"

echo "========================================================================"
echo "SHR CHECKPOINT AGGREGATION"
echo "========================================================================"
echo "Checkpoint folder: ${CHECKPOINT_FOLDER}"
echo "Base filename:     ${PRED_FILENAME}"
echo "Output path:       ${OUT_PATH}"
echo "========================================================================"
echo ""

echo "Checkpoint files found:"
ls -lh "${CHECKPOINT_FOLDER}"/*checkpoint*.pkl 2>/dev/null
echo ""

python /burg-archive/home/al4385/phenofusion/src/phenofusion/dataio/aggregate_pred_checkpoints.py \
    --pred_folder "${CHECKPOINT_FOLDER}" \
    --pred_filename "${PRED_FILENAME}" \
    --out "${OUT_PATH}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "AGGREGATION COMPLETE"
    echo "========================================================================"
    ls -lh "${OUT_PATH}"
else
    echo ""
    echo "========================================================================"
    echo "AGGREGATION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
