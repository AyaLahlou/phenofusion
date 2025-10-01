#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=pred_bdt_20_60   # The job name
#SBATCH --output=pred_bdt_20_60.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH -C mem768


module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


# _______________Predict on 10 test data_______________

PFT="BDT_20_60"


#july 2025
DATA_DIR="/burg/glab/users/al4385/data/TFT_40_overlapping_samples/sorted_BDT_-20_-60_1982_2021.pkl"
WEIGHTS_DIR="/burg/glab/users/al4385/weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth"
PRED_DIR="/burg/glab/users/al4385/predictions/pred_40year_moresamples/BDT/BDT_-20_-60_1982_2021.pkl"



python /burg/home/al4385/code/predict_tft.py  --data_dir $DATA_DIR --weights_dir $WEIGHTS_DIR --pred_dir $PRED_DIR
