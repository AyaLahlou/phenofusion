#!/bin/sh
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=driv_GRA   # The job name
#SBATCH --output=driver_GRA.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=160G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM
#SBATCH -C mem768

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch
pip install cartopy


PFT="GRA"
DATE=$(date '+%Y%m%d')

DATA_DIR="/burg/glab/users/al4385/data/TFT_40_overlapping_samples/${PFT}_1982_2021.pkl"
PRED_DIR="/burg/glab/users/al4385/predictions/pred_40year_moresamples/GRA/${PFT}_20251024.pkl"
COORD_DIR="/burg/glab/users/al4385/data/coordinates/${PFT}.parquet"
OUTPUT_DIR="/burg/glab/users/ms7073/analysis/driversdata/oversampling/${PFT}_${DATE}"
FORECAST_WINDOW=30

# generate drivers data
python /burg-archive/home/$USER/phenofusion/src/phenofusion/dataio/driversdata.py --PFT $PFT --pred_path $PRED_DIR --data_path $DATA_DIR --coord_path $COORD_DIR --output_path $OUTPUT_DIR --forecast_window_length $FORECAST_WINDOW

# generate driver maps
#MAP_DATA_DIR="/burg/glab/users/ms7073/analysis/driversdata/oversampling/"
#MAP_OUTPUT_DIR="/burg/glab/users/ms7073/analysis/driversdata/driver_maps_oversampling/"
#python /burg-archive/home/$USER/phenofusion/src/phenofusion/analysis/run_driver_maps.py --data-dir $MAP_DATA_DIR --output-dir $MAP_OUTPUT_DIR --show-plots
