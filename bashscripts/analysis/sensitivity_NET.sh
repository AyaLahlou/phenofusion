#!/bin/sh
#SBATCH --account=glab
#SBATCH --job-name=sens_NET
#SBATCH --output=sensitivity_NET_%j.out
#SBATCH --error=sensitivity_NET_%j.err
#SBATCH -c 2
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-12:00
#SBATCH -C mem768

module load anaconda

OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/data/sensitivity_data"
mkdir -p "$OUTPUT_DIR"

python /burg-archive/home/al4385/phenofusion/scripts/extract_epoch_drivers.py \
  --pft NET \
  --data-path /burg-archive/glab/users/al4385/data/TFT_40_overlapping_samples/NET_1982_2021.pkl \
  --pred-path /burg-archive/glab/users/al4385/predictions/pred_40year_moresamples/NET_1982_2021.pkl \
  --coords-path /burg-archive/glab/users/al4385/data/coordinates/NET.parquet \
  --sos-indices /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NET/SOS_pred_indices_NET.npy \
  --eos-indices /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/NET/EOS_pred_indices_NET.npy \
  --output-dir "$OUTPUT_DIR"
