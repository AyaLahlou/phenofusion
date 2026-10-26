#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_nodes
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --time=0-00:30
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
/burg-archive/glab/users/al4385/envs/retrain2026/bin/python scripts/review/item6/find_nodes.py /burg-archive/glab/users/al4385/data/CSIFMETEO_smfix/NDT_merged/sorted_NDT_merged_1982_2021.parquet
