#!/bin/sh
#SBATCH --account=glab
#SBATCH --job-name=sens_BET
#SBATCH --output=sensitivity_BET_%j.out
#SBATCH --error=sensitivity_BET_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=200G
#SBATCH --time=0-12:00
#SBATCH -C mem768

module load anaconda

OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
mkdir -p "$OUTPUT_DIR"

# BET SOS/EOS index files are empty -- use CSIF-slope detection instead
python /burg-archive/home/al4385/phenofusion/scripts/extract_epoch_drivers.py \
  --pft BET \
  --data-path /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/BET_1982_2021.pkl \
  --pred-path /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/BET_1982_2021.pkl \
  --coords-path /burg-archive/glab/users/al4385/data/coordinates/BET.parquet \
  --detect-csif \
  --output-dir "$OUTPUT_DIR"
