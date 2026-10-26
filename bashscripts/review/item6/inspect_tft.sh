#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_inspect_tft
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 2
#SBATCH --mem=80G
#SBATCH --time=0-01:00
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
PY=/burg-archive/glab/users/al4385/envs/retrain2026/bin/python
D=/burg-archive/glab/users/al4385/data
$PY scripts/review/item6/inspect_tft.py $D/TFT_30/NDT.pickle $D/TFT_smfix_2026/NDT.pkl
