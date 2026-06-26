#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=driver_map_BDT   # The job name
#SBATCH --output=driver_map_BDT.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=32G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


# _______________Predict on 10 test data_______________

PFT="BDT"

DATE=$(date '+%Y%m%d')

# --- Directories containing the chunked files ---
DATA_DIR="/burg/glab/users/al4385/data/TFT_30_40years/"
PRED_DIR="/burg/glab/users/al4385/predictions/TFT_30_40years/"
COORD_DIR="/burg/glab/users/al4385/data/coordinates/"

# --- Output and phenology paths ---
MAP_DIR="/burg/home/al4385/figures/drivermaps/${DATE}/${PFT}/"
PHENOLOGY_PATH="/burg/home/al4385/code/${PFT}_phenology_pixel.parquet"

# Create output directory if it doesn't exist
mkdir -p $MAP_DIR

PRED_FILES="pred_BDT_-20_20.pkl pred_BDT_-20_-60.pkl pred_BDT_50_20.pkl pred_BDT_50_90.pkl"
DATA_FILES="BDT_-20_20.pickle BDT_-20_-60.pickle BDT_50_20.pickle BDT_50_90.pickle"
COORD_FILES="BDT_-20_20.parquet BDT_-20_-60.parquet BDT_50_20.parquet BDT_50_90.parquet"


# --- Execute the Python script with all files ---
python attention_drivers_map_v2_parts.py \
    --pft ${PFT} \
    --output_path ${MAP_DIR} \
    --phenology_path ${PHENOLOGY_PATH} \
    --pred_dir ${PRED_DIR} \
    --data_dir ${DATA_DIR} \
    --coord_dir ${COORD_DIR} \
    --pred_list ${PRED_FILES} \
    --data_list ${DATA_FILES} \
    --coord_list ${COORD_FILES}
