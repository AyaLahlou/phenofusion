#!/bin/sh
#
# Create Analysis DataFrame from TFT Predictions
#
#SBATCH --account=glab
#SBATCH --job-name=create_analysisdf
#SBATCH --output=create_analysisdf_%j.out
#SBATCH -c 2
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-4:00
#SBATCH --qos=hpc_test
#SBATCH -C mem768

module load anaconda

# Set LD_LIBRARY_PATH to use Anaconda's libstdc++ which has the required GLIBCXX version
export LD_LIBRARY_PATH=/burg/opt/anaconda3-2023.09/lib:$LD_LIBRARY_PATH

# Configuration
PFT="BDT"  # Change this to process different PFTs
START_YEAR=1982
END_YEAR=2021

# File paths
DATA_DIR="/burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl"
PRED_DIR="/burg/glab/users/al4385/predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl"
COORD_DIR="/burg/glab/users/al4385/data/coordinates/merged_BDT.parquet"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/analysis_dataframes/"

# Output file
OUTPUT_FILE="${OUTPUT_DIR}/${PFT}_analysisdf.csv"

echo "=================================================="
echo "Creating Analysis DataFrame"
echo "=================================================="
echo "PFT: $PFT"
echo "Years: $START_YEAR - $END_YEAR"
echo "Output: $OUTPUT_FILE"
echo "=================================================="

# Run the script
python /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/create_analysis_dataframe.py \
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
