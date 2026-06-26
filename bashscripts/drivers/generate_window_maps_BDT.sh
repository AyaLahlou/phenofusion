#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=window_maps_BDT_%j.out
#SBATCH --error=window_maps_BDT_%j.err
#SBATCH --job-name=window_maps_BDT
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0-1:00
#SBATCH --partition=short

# Generate attention window maps for BDT
# Usage: sbatch bashscripts/drivers/generate_window_maps_BDT.sh

module purge
module load anaconda

DATA_DIR="/burg-archive/home/al4385/phenofusion/data/attention_window"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/attention_window/maps"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================================"
echo "WINDOW MAP GENERATION - BDT"
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

echo "Starting window map generation..."
python /burg-archive/home/al4385/phenofusion/scripts/generate_window_maps.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --projection Robinson \
    --cmap plasma \
    --dpi 300

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "WINDOW MAP GENERATION COMPLETE"
    echo "========================================================================"
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}"/*.png 2>/dev/null || echo "  (no PNG files found)"
else
    echo ""
    echo "========================================================================"
    echo "WINDOW MAP GENERATION FAILED (exit code: $exit_code)"
    echo "========================================================================"
fi

exit $exit_code
