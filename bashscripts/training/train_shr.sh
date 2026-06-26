#!/bin/sh
#
# Training script for SCH (Shrubland) vegetation type using TFT with transfer learning
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=SHR_TL     # The job name
#SBATCH --output=SHR_TL.out   # The output file name
#SBATCH --gres=gpu:1
#SBATCH -c 1                  # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=4-23:00        # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=126gb   # The memory the job will use per cpu core

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0"


module load anaconda
module load cuda11.8/toolkit
nvcc --version

# Check CUDA version using nvidia-smi
#Command to execute Python program
pip install tft-torch

# Set paths
CHECKPOINT_PATH="/burg/glab/users/al4385/weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth"
FILENAME="sorted_SHR.pkl"
DATA_DIR="/burg/glab/users/al4385/data/TFT_30/"
WEIGHTS_DIR="/burg/glab/users/al4385/weights/pretrained_1219/"
WANDB_PROJECT="Transfer_Learning_SHR_freeze"

# Train SHR vegetation type with transfer learning
python /burg-archive/home/al4385/phenofusion/src/phenofusion/models/train_tft_TL.py \
    --filename "$FILENAME" \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_directory "$DATA_DIR" \
    --weights_directory "$WEIGHTS_DIR" \
    --wandb_project "$WANDB_PROJECT"
