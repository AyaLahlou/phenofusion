#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=r6_inspect
#SBATCH --output=outputs/review/item6/logs/%x_%j.out
#SBATCH -c 2
#SBATCH --mem=60G
#SBATCH --time=0-01:00
set -euo pipefail
cd /burg-archive/home/al4385/phenofusion
export PYTHONNOUSERSITE=1
PY=/burg-archive/glab/users/al4385/envs/retrain2026/bin/python
D=/burg-archive/glab/users/al4385/data
$PY scripts/review/item6/inspect_benchmarks.py $D/4GST/NDT_predictions.pkl $D/phenoformer/NDT_predictions.pkl $D/pyphenology_benchmark/BET_ThermalTime_predictions.pkl $D/4GST/BET_predictions.pkl $D/pyphenology_formatted/NDT_test_pyphenology.pkl
