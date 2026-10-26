#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_pcmodels
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=0-03:00
# PhenoCam vs TFT / PhenoFormerSeq / 4GST dates at PhenoCam BDT pixels (phenocam_models.py); needs phenocam_cache.sh first
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
/burg-archive/glab/users/al4385/envs/retrain2026/bin/python -W ignore scripts/review/item6/phenocam_models.py "$@"
