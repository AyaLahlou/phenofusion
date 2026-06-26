#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=a40bdt_map   # The job name
#SBATCH --output=a40bdt_map.out    # The output file name
#SBATCH -c 4                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=32G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH --constraint=a40         # Specify node constraint for v100s GPUs
#SBATCH --nodelist=g184            # Specify the node to run the job


module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


# _______________Predict on 10 test data_______________

PFT="merged_BDT"

DATE=$(date '+%Y%m%d')
DATA_DIR="/burg/glab/users/al4385/data/TFT_30/${PFT}_1982_2021.pkl"
WEIGHTS_DIR="/burg/glab/users/al4385/weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth"
PRED_DIR="/burg/glab/users/al4385/predictions/pretrained_1219/pred_${PFT}_2_jan2362025.pkl"

#python predict_tft.py  --data_dir $DATA_DIR --weights_dir $WEIGHTS_DIR --pred_dir $PRED_DIR

# _______________________Metrics_______________________
METRICS_FIG_DIR="/burg/home/al4385/figures/TFT_pretrained_metrics/${DATE}/"
mkdir -p $METRICS_FIG_DIR
#python metrics_tft.py --filename "pred_${PFT}_2_jan232025.pkl" --data_dir $DATA_DIR --pred_dir $PRED_DIR --fig_dir "${METRICS_FIG_DIR}metrics_${PFT}_2_jan232025.png"

# Predict on 40 years of data
DATA_DIR_40="/burg/glab/users/al4385/data/TFT_30_40years/sorted_${PFT}_1982_2021.pkl"
PRED_DIR_40="/burg/glab/users/al4385/predictions/TFT_pretrained_40y/${DATE}/pred_${PFT}_mar032025.pkl"
mkdir -p $PRED_DIR_40
python predict_tft.py  --data_dir $DATA_DIR_40 --weights_dir $WEIGHTS_DIR --pred_dir "${PRED_DIR_40}pred_${PFT}.pkl"

# attention map
MAP_DIR="/burg/home/al4385/figures/drivermaps/${DATE}/${PFT}/"
mkdir -p $MAP_DIR
COORD_DIR="/burg/glab/users/al4385/data/coordinates/${PFT}.parquet"
CLUST_LIST=("${PFT}" )
python attention_drivers_map.py --pred_path $PRED_DIR_40 --data_path $DATA_DIR_40 --coord_path $COORD_DIR --output_path $MAP_DIR --cluster_list "${CLUST_LIST[@]}"
#
