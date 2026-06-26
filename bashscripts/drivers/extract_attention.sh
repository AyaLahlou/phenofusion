#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=extract_attention
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-01:00
#SBATCH -C mem768

# Load required modules
module load anaconda

# Run the extraction script
cd /burg-archive/home/al4385/phenofusion
python extract_attention_scores.py
