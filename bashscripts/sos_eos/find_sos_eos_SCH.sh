#!/bin/sh
#
# Find SOS/EOS from Mean Seasonal Cycle for BDT
#
#SBATCH --account=glab
#SBATCH --job-name=sos_eos_SCH
#SBATCH --output=sos_eos_SCH_%j.out
#SBATCH -c 2
#SBATCH --mem-per-cpu=130G
#SBATCH --time=0-2:00



module load anaconda
module load gcc/11.2.0

# Set LD_LIBRARY_PATH to use Anaconda's libstdc++ which has the required GLIBCXX version
export LD_LIBRARY_PATH=/burg/opt/anaconda3-2023.09/lib:$LD_LIBRARY_PATH

# Input and output paths
PFT="SCH"
INPUT_FILE="/burg-archive/glab/users/al4385/data/CSIFMETEO/SCH_merged.parquet"
OUTPUT_JSON="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_subset_data/sos_eos_SCH_merged_smoothed.json"
FIGURES_DIR="/burg-archive/home/al4385/phenofusion/debugging_data_artifacts/debugging_figures/sos_eos_ranges/SCH"

# Run the script
python /burg-archive/home/al4385/phenofusion/debugging_data_artifacts/find_sos_eos_from_msc_notBDT.py \
    --input $INPUT_FILE \
    --output $OUTPUT_JSON \
    --figures-dir $FIGURES_DIR \
    --sos-thresholds 0.15 0.40 \
    --eos-thresholds 0.70 0.45

echo "SOS/EOS detection complete!"
echo "Output saved to: $OUTPUT_JSON"
echo "Figures saved to: $FIGURES_DIR"
