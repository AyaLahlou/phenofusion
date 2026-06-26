#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=agg_check   # The job name
#SBATCH --output=agg_check.out    # The output file name
#SBATCH -c 4                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH -C mem768


python /burg-archive/home/al4385/phenofusion/src/phenofusion/dataio/aggregate_pred_checkpoints.py \
  --pred_folder /burg/glab/users/al4385/predictions/pred_40year_moresamples/BDT \
  --pred_filename BDT_-20_20_1982_2021.pkl \
  --out /burg/glab/users/al4385/predictions/pred_40year_moresamples/BDT_-20_20_1982_2021.pkl
