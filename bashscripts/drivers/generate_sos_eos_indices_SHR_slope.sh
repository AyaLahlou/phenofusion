#!/bin/bash
#SBATCH --account=glab
#SBATCH --output=sos_eos_indices_SHR_slope_%j.out
#SBATCH --error=sos_eos_indices_SHR_slope_%j.err
#SBATCH --job-name=sos_eos_SHR_slope
#SBATCH --mem-per-cpu=64G
#SBATCH --time=0-2:00
#SBATCH --partition=short

module purge
module load anaconda

echo "========================================================================"
echo "SOS/EOS INDEX GENERATION (slope-based) - SHR"
echo "========================================================================"

python /burg-archive/home/al4385/phenofusion/debugging/save_sos_eos_indices_SHR_slope.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "COMPLETE"
    ls -lh /burg-archive/home/al4385/phenofusion/data/SOS_EOS_indices/SHR/*.npy
else
    echo "FAILED (exit code: $exit_code)"
fi

exit $exit_code
