#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=driver_data   # The job name
#SBATCH --output=driver_data.out    # The output file name
#SBATCH -c 2                   # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --mem-per-cpu=32G        # The memory the job will use per cpu core
#SBATCH --time=0-15:00            # The time the job will take to run in D-HH:MM

module load anaconda
module load cuda11.1/toolkit
pip install tft-torch


#
DATE=$(date '+%Y%m%d')

if [[ $1 == "BDT" ]]; then
    PRED_FILES=("pred_BDT_-20_20_merged_1982_2021.pkl" "pred_BDT_-20_-60_1982_2021.pkl" "pred_BDT_50_20_merged_1982_2021.pkl" "sorted_BDT_50_90_1982_2021.pkl")
    DATA_FILES=("sorted_BDT_-20_20_merged_1982_2021.pkl" "sorted_BDT_-20_-60_1982_2021.pkl" "sorted_BDT_50_20_merged_1982_2021.pkl" "sorted_BDT_50_90_1982_2021.pkl")
    COORD_FILES=("BDT_-20_20.parquet" "BDT_-20_-60.parquet" "BDT_50_20.parquet" "BDT_50_90.parquet")

    for (( i=0; i<4; i++ )); do
        # grab element and build directory name
        DATA_DIR="/burg/glab/users/al4385/data/TFT_30_40_BDTinterval/${DATA_FILES[i]}"
        PRED_DIR="/burg/glab/users/al4385/predictions/pretrained_0331/${PRED_FILES[i]}"
        COORD_DIR="/burg/glab/users/al4385/data/coordinates/${COORD_FILES[i]}"
        name="$(basename "${COORD_FILES[i]}" .parquet)"
        OUT_DIR="/burg/home/al4385/code/phenology_analysis/drivers_data/${name}_${DATE}"
        OUT_DIR="/burg/home/al4385/code/phenology_analysis/df_analysis/${name}_${DATE}"

        echo "Index $i → data_dir=$DATA_DIR → pred_dir=$PRED_DIR → coord_dir=$COORD_DIR"
        python attention_drivers_map_v2.py --PFT "$1" --pred_path "$PRED_DIR" --data_path "$DATA_DIR" --coord_path "$COORD_DIR" --output_path "$OUT_DIR"
    done

else
    PFT="$1"
    DATA_DIR="/burg/glab/users/al4385/data/TFT_30_40years/${PFT}.pickle"
    PRED_DIR="/burg/glab/users/al4385/predictions/predictions_40years/${PFT}/preds.pickle"
    OUT_DIR="/burg/home/al4385/code/phenology_analysis/drivers_data/${PFT}_${DATE}"
    OUT_DIR="/burg/home/al4385/code/phenology_analysis/df_analysis/${PFT}_${DATE}"
    COORD_DIR="/burg/glab/users/al4385/data/coordinates/${PFT}.parquet"

    python attention_drivers_map_v2.py --PFT "$PFT" --pred_path "$PRED_DIR" --data_path "$DATA_DIR" --coord_path "$COORD_DIR" --output_path "$OUT_DIR"

fi
