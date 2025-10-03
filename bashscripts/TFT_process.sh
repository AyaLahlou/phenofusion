#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=tftprocess    # The job name
#SBATCH --output=v100process_mar14.out    # The output file name
#SBATCH -c 1
#SBATCH --time=0-20:00            # The time the job will take to run in D-HH:MM
#SBATCH -C mem768


module load anaconda
module load cuda11.1/toolkit

#printf "Hello World! I am process %d of %d\n" $SLURM_PROCID $SLURM_NTASKS

# Command to execute Python program
#python TFT_process.py
python TFT_process.py
# End of script
