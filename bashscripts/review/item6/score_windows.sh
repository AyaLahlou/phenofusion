#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_windows
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 2
#SBATCH --mem=120G
#SBATCH --time=0-03:00
# Window-level benchmark scoring (score_windows.py). Usage: sbatch [--mem=170G] score_windows.sh PFT
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
/burg-archive/glab/users/al4385/envs/retrain2026/bin/python -W ignore scripts/review/item6/score_windows.py --pft "$1"
