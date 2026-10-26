#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_sim
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --time=0-01:00
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
/burg-archive/glab/users/al4385/envs/retrain2026/bin/python -W ignore scripts/review/item6/sim_attenuation.py
