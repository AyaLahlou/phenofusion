#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=mean_attention_maps_%j.out
#SBATCH --error=mean_attention_maps_%j.err
#SBATCH --job-name=mean_attention_maps
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-1:00
#SBATCH --partition=short

# Generate attention-weighted mean day maps for all PFTs
# Usage: sbatch bashscripts/drivers/generate_mean_attention_maps.sh

module purge
module load anaconda

DATA_DIR="/burg-archive/home/al4385/phenofusion/data/attention_mean"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/attention_mean/maps"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "MEAN ATTENTION DAY MAP GENERATION"
echo "========================================================================"
echo "Data dir:   ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""

echo "Checking input files..."
for file in "${DATA_DIR}/BDT_SOS.csv" "${DATA_DIR}/BDT_EOS.csv"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
    echo "  Found: $(basename $file)"
done
echo ""

echo "Starting mean attention day map generation..."
python /burg-archive/home/al4385/phenofusion/scripts/generate_mean_attention_maps.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --projection Robinson \
    --cmap tab20_no_blue \
    --dpi 300 \
    --crop-lat-min -60 \
    --crop-lat-max 71 \
    --crop-lon-min -160 \
    --crop-lon-max 165

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "MEAN ATTENTION DAY MAP GENERATION COMPLETE"
    echo "========================================================================"
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"/*.png 2>/dev/null || echo "  (no PNG files found)"
else
    echo ""
    echo "========================================================================"
    echo "MEAN ATTENTION DAY MAP GENERATION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
