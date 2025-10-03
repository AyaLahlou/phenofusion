#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=glab        # Replace ACCOUNT with your group account name
#SBATCH --job-name=BET    # The job name
#SBATCH --output=BETfeb17.out    # The output file name
#SBATCH --gres=gpu:1
#SBATCH -c 1                    # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=4-23:00            # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=126gb         # The memory the job will use per cpu core
#SBATCH --constraint=v100s         # Specify node constraint for v100s GPUs
#SBATCH --nodelist=g050            # Specify the node to run the job

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0"


module load anaconda
module load cuda11.8/toolkit
nvcc --version

# Check CUDA version using nvidia-smi
#Command to execute Python program
pip install tft-torch

# Set the GPU to use (e.g., GPU 0)

#first train boreal 2
CHECKPOINT_PATH="/burg/glab/users/al4385/weights/pretrained_1219/weights_merged_BDT_1982_2021_feb2025_checkpoint.pth"
python train_tft_TL.py BET.pickle --checkpoint $CHECKPOINT_PATH



#python train_tft.py sorted_BoND_1982_2021.pkl
#python train_tft.py sorted_BoNE_1982_2021.pkl
#python train_tft.py merged_BDT_1982_2021.pkl
#python train_tft.py sorted_Sav_1982_2021.pkl

# second train boreal 2
#python train_tft.py sorted_Sch_1982_2021.pkl
#python train_tft.py sorted_TeBD_1982_2021.pkl
#python train_tft.py sorted_TeBE_1982_2021.pkl
#python train_tft.py sorted_TeNE_1982_2021.pkl

# third train boreal 2
#python train_tft.py sorted_TrBD_1982_2021.pkl
#python train_tft.py sorted_TrBE_1982_2021.pkl
#python train_tft.py sorted_Tun_1982_2021.pkl


# End of script
