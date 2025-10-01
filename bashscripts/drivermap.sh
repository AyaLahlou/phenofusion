#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=driver_map   # The job name
#SBATCH --output=driver_map.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH -C mem768

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


# _______________Predict on 10 test data_______________

PFT="BDT"

DATE=$(date '+%Y%m%d')
DATA_DIR="/burg/glab/users/al4385/data/TFT_30_40_BDTinterval/"
PRED_DIR="/burg/glab/users/al4385/predictions/pretrained_0331/"

# attention map
MAP_DIR="/burg/home/al4385/figures/drivermaps/${DATE}/${PFT}/"
mkdir -p $MAP_DIR
COORD_DIR="/burg/glab/users/al4385/data/coordinates/"
#list of cluster filenames
COORD_LIST=("BDT_-20_20.parquet" "BDT_-20_-60.parquet" "BDT_50_20.parquet" "BDT_50_90.parquet")
#list of data filenames
DATA_LIST=("sorted_BDT_-20_20_merged_1982_2021.pkl" "sorted_BDT_-20_-60_1982_2021.pkl" "sorted_BDT_50_20_merged_1982_2021.pkl" "sorted_BDT_50_90_1982_2021.pkl")
#list of prediction filenames
PRED_LIST=("pred_BDT_-20_20_merged_1982_2021.pkl"  "pred_BDT_-20_-60_1982_2021.pkl"  "pred_BDT_50_20_merged_1982_2021.pkl"  "sorted_BDT_50_90_1982_2021.pkl")



python attention_drivers_map_parts.py --pred_path $PRED_DIR --data_path $DATA_DIR --coord_path $COORD_DIR --output_path $MAP_DIR --coord_list $COORD_LIST --data_list $DATA_LIST --pred_list $PRED_LIST
#
