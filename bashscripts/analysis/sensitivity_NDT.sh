#!/bin/sh
#SBATCH --account=glab
#SBATCH --job-name=sens_NDT
#SBATCH --output=sensitivity_NDT_%j.out
#SBATCH --error=sensitivity_NDT_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=80G
#SBATCH --time=0-08:00

module load anaconda

OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
mkdir -p "$OUTPUT_DIR"

python /burg-archive/home/al4385/phenofusion/scripts/extract_epoch_drivers.py \
  --pft NDT \
  --data-path /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/NDT_1982_2021.pkl \
  --pred-path /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/NDT_1982_2021.pkl \
  --coords-path /burg-archive/glab/users/al4385/data/coordinates/NDT.parquet \
  --sos-indices /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/SOS_pred_indices_NDT.npy \
  --eos-indices /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NDT/EOS_pred_indices_NDT.npy \
  --output-dir "$OUTPUT_DIR"
