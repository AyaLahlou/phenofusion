#!/bin/sh
#SBATCH --account=glab
#SBATCH --job-name=sens_SHR
#SBATCH --output=sensitivity_SHR_%j.out
#SBATCH --error=sensitivity_SHR_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=0-08:00

module load anaconda

OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
mkdir -p "$OUTPUT_DIR"

python /burg-archive/home/al4385/phenofusion/scripts/extract_epoch_drivers.py \
  --pft SHR \
  --data-path /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/SHR_1982_2021.pkl \
  --pred-path /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/SHR_20251023.pkl \
  --coords-path /burg-archive/glab/users/al4385/data/coordinates/SHR.parquet \
  --sos-indices /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/SHR/SOS_pred_indices_SHR.npy \
  --eos-indices /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/SHR/EOS_pred_indices_SHR.npy \
  --output-dir "$OUTPUT_DIR"
