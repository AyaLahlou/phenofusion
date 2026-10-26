#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_pccache
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 2
#SBATCH --mem=170G
#SBATCH --time=0-03:00
# PhenoCam model comparison, step A: cache scalers/feature map + validation windows (phenocam_cache.py)
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
/burg-archive/glab/users/al4385/envs/retrain2026/bin/python -W ignore scripts/review/item6/phenocam_cache.py
