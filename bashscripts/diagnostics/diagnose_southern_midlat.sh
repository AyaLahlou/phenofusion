#!/bin/sh
#
# Diagnose Southern Hemisphere Mid-Latitude SOS/EOS Detection
#
#SBATCH --account=glab
#SBATCH --job-name=diag_south_mid
#SBATCH --output=diagnose_southern_midlat_%j.out
#SBATCH -c 2
#SBATCH --mem-per-cpu=130G
#SBATCH --time=0-2:00
#SBATCH --partition=short


module load anaconda
module load gcc/11.2.0

# Set LD_LIBRARY_PATH to use Anaconda's libstdc++ which has the required GLIBCXX version
export LD_LIBRARY_PATH=/burg/opt/anaconda3-2023.09/lib:$LD_LIBRARY_PATH

# Input and output paths
INPUT_FILE="/burg-archive/glab/users/al4385/data/CSIFMETEO/sorted_merged_BDT_1982_2021.parquet"
OUTPUT_DIR="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_figures/southern_midlat"

# Run the diagnostic script
python /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/diagnose_southern_midlat.py \
    --input $INPUT_FILE \
    --output-dir $OUTPUT_DIR \
    --lat-min -42 \
    --lat-max -25 \
    --sigma 2.5 \
    --sos-thresholds 0.15 0.40 \
    --eos-thresholds 0.85 0.60

echo "Diagnostic complete!"
echo "Results saved to: $OUTPUT_DIR"
