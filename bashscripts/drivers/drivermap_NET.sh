#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=driver_map_NET   # The job name
#SBATCH --output=driver_map_NET.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=32G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


# _______________Predict on 10 test data_______________

PFT="NET"

DATE=$(date '+%Y%m%d')
DATA_DIR="/burg/glab/users/al4385/data/TFT_30_40years/${PFT}.pickle"
PRED_DIR="/burg/glab/users/al4385/predictions/predictions_40years/${PFT}/preds.pickle"
MAP_DIR="/burg/home/al4385/figures/drivermaps/${DATE}/${PFT}/"
COORD_DIR="/burg/glab/users/al4385/data/coordinates/${PFT}.parquet"
PHENOLOGY_DICT_PATH="/burg/home/al4385/code/${PFT}_phenology_pixel.parquet" # Make sure this path is correct

mkdir -p $MAP_DIR

python attention_drivers_map_v2.py --pred_path $PRED_DIR --data_path $DATA_DIR --coord_path $COORD_DIR --output_path $MAP_DIR --phenology_path $PHENOLOGY_DICT_PATH
