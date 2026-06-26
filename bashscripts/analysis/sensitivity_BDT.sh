#!/bin/sh
#SBATCH --account=glab
#SBATCH --job-name=sens_BDT
#SBATCH --output=sensitivity_BDT_%j.out
#SBATCH --error=sensitivity_BDT_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=200G
#SBATCH --time=0-12:00
#SBATCH -C mem768

module load anaconda

OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
mkdir -p "$OUTPUT_DIR"

python /burg-archive/home/al4385/phenofusion/scripts/extract_epoch_drivers.py \
  --pft BDT \
  --data-path /burg-archive/glab/users/al4385/data/TFT_30_40years/sorted_merged_BDT_1982_2021.pkl \
  --pred-path /burg-archive/glab/users/al4385/predictions/predictions_40years/BDT/sorted_merged_BDT_1982_2021_TFT30_40years.pkl \
  --coords-path /burg-archive/glab/users/al4385/data/coordinates/merged_BDT.parquet \
  --sos-indices /burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/SOS_pred_indices_BDT_1.npy \
  --eos-indices /burg-archive/home/al4385/phenofusion/debugging/debugging_subset_data/EOS_pred_indices_BDT_1.npy \
  --output-dir "$OUTPUT_DIR"
