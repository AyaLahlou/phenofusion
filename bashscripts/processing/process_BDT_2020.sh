#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=process    # The job name
#SBATCH --output=process_2020_bdt.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-24:00            # The time the job will take to run in D-HH:MM

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


PFT="BDT_20_20"
DATA_DIR="/burg/glab/users/al4385/data/CSIFMETEO/sorted_BDT_-20_20_merged_1982_2021.parquet"
OUT_DIR="/burg/glab/users/al4385/data/TFT_40_overlapping_samples/"



python /burg/home/al4385/code/TFT_process.py --PFT $PFT --data_path $DATA_DIR --output_path $OUT_DIR
