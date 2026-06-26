#!/bin/sh
#
# Extract SOS/EOS indices from analysis dataframes for BDT
#
#SBATCH --account=glab
#SBATCH --job-name=save_indices_NDT
#SBATCH --output=save_indices_NDT_%j.out
#SBATCH -c 2
#SBATCH --mem-per-cpu=160G
#SBATCH --time=0-2:00




module load anaconda
module load gcc/11.2.0

# Set LD_LIBRARY_PATH to use Anaconda's libstdc++ which has the required GLIBCXX version
export LD_LIBRARY_PATH=/burg/opt/anaconda3-2023.09/lib:$LD_LIBRARY_PATH

# Run the script
python /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/save_sos_eos_indices.py

echo "SOS/EOS indices extraction complete!"
echo "Output files saved to: /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_subset_data/"
echo "Figures saved to: /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_figures/"
