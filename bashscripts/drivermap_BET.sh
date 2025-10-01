#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=driver_data   # The job name
#SBATCH --output=driver_data.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=32G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


#
DATE=$(date '+%Y%m%d')

PFT="BET"
DATA_DIR="/burg/glab/users/al4385/data/TFT_30_40years/${PFT}.pickle"
PRED_DIR="/burg/glab/users/al4385/predictions/predictions_40years/${PFT}/preds.pickle"
OUT_DIR="/burg/home/al4385/code/phenology_analysis/drivers_data/${PFT}_${DATE}"
COORD_DIR="/burg/glab/users/al4385/data/coordinates/${PFT}.parquet"

python attention_drivers_map_BET.py --pred_path "$PRED_DIR" --data_path "$DATA_DIR" --coord_path "$COORD_DIR" --output_path "$OUT_DIR"
