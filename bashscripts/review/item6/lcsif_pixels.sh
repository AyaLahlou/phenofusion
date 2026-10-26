#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_pix
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH --time=0-04:00
# LCSIF dates (smooth/node/ev2) + daily T for a pixel set. Usage: sbatch lcsif_pixels.sh PFT TAG [extra args]
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
P=$1; TAG=$2; shift 2
/burg-archive/glab/users/al4385/envs/retrain2026/bin/python -W ignore scripts/review/item6/lcsif_pixels.py --pft $P --tag $TAG "$@"
