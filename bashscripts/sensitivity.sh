#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=sensitivity   # The job name
#SBATCH --output=sensitivity.out   # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH -C mem768


module load anaconda
module load cuda11.1/toolkit
pip install tft-torch

DATA_DIR="/burg/glab/users/al4385/data/TFT_40_overlapping_samples/"
PRED_DIR="/burg/glab/users/al4385/predictions/pred_40year_moresamples/"
COORD_DIR="/burg/glab/users/al4385/data/coordinates/"
OUTPUT_DIR="/burg/glab/users/ms7073/analysis/sensitivity_analysis_oversampling/"


python /burg-archive/home/$USER/phenofusion/src/phenofusion/analysis/run_sensitivity_analysis.py --data-dir $DATA_DIR --pred-dir $PRED_DIR --coord-dir $COORD_DIR --output-dir $OUTPUT_DIR --show-plots
