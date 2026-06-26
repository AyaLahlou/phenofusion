#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=drivers_BDT_json
#SBATCH --nodes=1
#SBATCH --output=drivers_BDT_json_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-6:00
#SBATCH -C mem768
#SBATCH --qos=hpc_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=al4385@columbia.edu

# Load required modules
module load anaconda

# Define paths
PFT="BDT"

DATA_PATH="/burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl"
PRED_PATH="/burg/glab/users/al4385/predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl"
COORDS_PATH="/burg/glab/users/al4385/data/coordinates/merged_BDT.parquet"
CONFIG_PATH="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_subset_data/sos_eos_BDT_merged_smoothed.json"
OUTPUT_PATH="/burg/glab/users/al4385/phenofusion/data/drivers_data/${PFT}_json_drivers"

# Print configuration
echo "=================================================="
echo "JSON-Based Driver Extraction for ${PFT}"
echo "=================================================="
echo "Data path: ${DATA_PATH}"
echo "Predictions path: ${PRED_PATH}"
echo "Coordinates path: ${COORDS_PATH}"
echo "Config path: ${CONFIG_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Forecast window: 30"
echo "Validate monotonic: No (use --validate_monotonic flag to enable)"
echo "=================================================="
echo ""

# Run the driver extraction script
python src/phenofusion/dataio/driversdata_maria_v2.py \
    --PFT ${PFT} \
    --pred_path ${PRED_PATH} \
    --data_path ${DATA_PATH} \
    --coord_path ${COORDS_PATH} \
    --config_path ${CONFIG_PATH} \
    --output_path ${OUTPUT_PATH} \
    --forecast_window_length 30

echo ""
echo "Job completed at $(date)"
