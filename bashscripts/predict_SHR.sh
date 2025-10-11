#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=pred_SHR   # The job name
#SBATCH --output=pred_SHR.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM


module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


PFT="SHR"


#oct 2025 DEV
DATA_DIR="/burg/glab/users/$USER/data/TFT_30/sorted_SHR.pkl"
WEIGHTS_DIR=/burg/glab/users/$USER/weights/pretrained_1219/weights_sorted_SHR_dev.pth
PRED_DIR="/burg/glab/users/$USER/predictions/pred_TFT_30_dev/SHR/SHR_1982_2021.pkl"


python /burg-archive/home/$USER/phenofusion/src/phenofusion/dataio/predict_tft.py  --data_dir $DATA_DIR --weights_dir $WEIGHTS_DIR --pred_dir $PRED_DIR
