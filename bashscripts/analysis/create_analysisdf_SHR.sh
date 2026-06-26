#!/bin/sh
#
# Create Analysis DataFrame from TFT Predictions for SHR
#
#SBATCH --account=glab
#SBATCH --job-name=create_analysisdf_SHR
#SBATCH --output=create_analysisdf_SHR_%j.out
#SBATCH -c 2
#SBATCH --mem-per-cpu=96G
#SBATCH --time=0-4:00

module load anaconda

# Set LD_LIBRARY_PATH to use Anaconda's libstdc++ which has the required GLIBCXX version
export LD_LIBRARY_PATH=/burg/opt/anaconda3-2023.09/lib:$LD_LIBRARY_PATH

# Configuration
PFT="SHR"
START_YEAR=1982
END_YEAR=2021

# File paths
# Note: SHR_1982_2021.pkl is truncated/corrupt; use SHR_20251023.pkl instead
DATA_DIR="/burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/SHR_1982_2021.pkl"
PRED_DIR="/burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/SHR_20251023.pkl"
COORD_DIR="/burg-archive/glab/users/al4385/data/coordinates/SHR.parquet"
OUTPUT_FILE="/burg-archive/glab/users/al4385/ms7073/analysis/driversdata/oversampling/SHR_analysisdf_20251106"

echo "=================================================="
echo "Creating Analysis DataFrame for SHR"
echo "=================================================="
echo "PFT: $PFT"
echo "Years: $START_YEAR - $END_YEAR"
echo "Data:  $DATA_DIR"
echo "Pred:  $PRED_DIR"
echo "Coord: $COORD_DIR"
echo "Output: $OUTPUT_FILE"
echo "=================================================="

# Run the script
python /burg-archive/home/al4385/phenofusion/debugging/create_analysis_dataframe.py \
    --pft $PFT \
    --data-dir $DATA_DIR \
    --pred-dir $PRED_DIR \
    --coord-dir $COORD_DIR \
    --start-year $START_YEAR \
    --end-year $END_YEAR \
    --output $OUTPUT_FILE \
    --verbose

echo ""
echo "=================================================="
echo "Analysis DataFrame creation complete!"
echo "Output saved to: $OUTPUT_FILE"
echo "=================================================="
