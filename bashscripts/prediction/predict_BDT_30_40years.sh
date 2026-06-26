#!/bin/sh
#
# Predict job for BDT 50-20 latitude band using TFT_30_40years data
#
#SBATCH --account=glab
#SBATCH --job-name=pred_bdt_30_40
#SBATCH --output=pred_bdt_30_40_%j.out
#SBATCH -c 2
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-06:00
#SBATCH --qos=hpc_test
#SBATCH -C mem768


module load anaconda
module load cuda11.1/toolkit
pip install tft-torch
pip install --upgrade --force-reinstall pyarrow


# _______________Predict on TFT_30_40years data_______________

PFT="merged_BDT"

TRAIN_DIR="/burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl"
DATA_DIR="/burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl"
WEIGHTS_DIR="/burg/glab/users/al4385/weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth"
PRED_DIR="/burg/glab/users/al4385/predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl"

# _______________Run prediction_______________

python /burg/home/al4385/code/predict_tft.py  --training_dir $TRAIN_DIR --data_dir $DATA_DIR --weights_dir $WEIGHTS_DIR --pred_dir $PRED_DIR
