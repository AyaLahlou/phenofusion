#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=process    # The job name
#SBATCH --output=proc_gra.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-24:00            # The time the job will take to run in D-HH:MM

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


PFT="GRA"
DATA_DIR="/burg-archive/glab/users/al4385/data/CSIFMETEO/sorted_GRA.parquet"



#OUT_DIR="/burg-archive/glab/users/al4385/data/TFT_30/"
#python /burg-archive/home/$USER/phenofusion/src/phenofusion/dataio/TFT_process.py  --data_path $DATA_DIR --output_path $OUT_DIR 


OUT_DIR="/burg/glab/users/al4385/data/TFT_40_overlapping_samples/"
python /burg-archive/home/$USER/phenofusion/src/phenofusion/dataio/TFT_process.py  --data_path $DATA_DIR --output_path $OUT_DIR --test_mode true --samp_interval 70
