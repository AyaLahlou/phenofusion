#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_download
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --time=0-06:00
# Downloads PhenoCam transition dates + USA-NPN site-level phenometrics (public, no credentials).
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
PY=/burg-archive/glab/users/al4385/envs/retrain2026/bin/python
$PY -I scripts/review/item6/download_ground.py "$@"
