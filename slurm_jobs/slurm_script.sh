#!/bin/bash
#SBATCH --job-name=pls_analysis
#SBATCH --time=36:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

# Function to check required variables
check_required_vars() {
    local missing=()
    for var in "$@"; do
        if [ -z "${!var}" ]; then
            missing+=("$var")
        fi
    done
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Error: Missing required environment variables: ${missing[*]}"
        exit 1
    fi
}

# Source config parameters
if [ ! -f "/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/config_params.sh" ]; then
    echo "Error: config_params.sh not found!"
    exit 1
fi
source /scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/config_params.sh

# Check required environment variables
required_vars=(
    "ROI"
    "CONFIG_SUBJECTS"
    "CONFIG_MOVIES"
    "CONFIG_METRIC"
    "CONFIG_ALIGNMENT"
    "CONFIG_PERFORM_PERMUTATIONS"
    "CONFIG_N_PERMUTATIONS"
    "CONFIG_PERFORM_DOWNSAMPLING"
    "CONFIG_N_DOWNSAMPLE"
    "CONFIG_RANDOM_STATE"
    "CONFIG_BLOCK_SIZE"
    "CONFIG_DATA_PATH"
)
check_required_vars "${required_vars[@]}"

# Load required modules
module load anaconda
conda activate indiv_dim

echo "Processing job for ROI: $ROI"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Get batch range
if [ ! -f "/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/batch_ranges.txt" ]; then
    echo "Error: batch_ranges.txt not found!"
    exit 1
fi

BATCH_RANGE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/batch_ranges.txt)
if [ -z "$BATCH_RANGE" ]; then
    echo "Error: Could not read batch range for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

START_IDX=$(echo $BATCH_RANGE | cut -d',' -f1)
END_IDX=$(echo $BATCH_RANGE | cut -d',' -f2)

echo "Processing batch $SLURM_ARRAY_TASK_ID (pairs $START_IDX to $END_IDX)"

# Run PLS analysis
if ! python /scratch4/mbonner5/chan21/indiv_dim/scripts/run_pls.py \
    --roi "$ROI" \
    --batch-start "$START_IDX" \
    --batch-end "$END_IDX" \
    --subjects "$CONFIG_SUBJECTS" \
    --metric "$CONFIG_METRIC" \
    --movies "$CONFIG_MOVIES" \
    --perform-permutations "$CONFIG_PERFORM_PERMUTATIONS" \
    --n-permutations "$CONFIG_N_PERMUTATIONS" \
    --perform-downsampling "$CONFIG_PERFORM_DOWNSAMPLING" \
    --n-downsample "$CONFIG_N_DOWNSAMPLE" \
    --alignment "$CONFIG_ALIGNMENT" \
    --random-state "$CONFIG_RANDOM_STATE" \
    --block-size "$CONFIG_BLOCK_SIZE" \
    --data-path "$CONFIG_DATA_PATH"; then
    
    echo "Error: PLS analysis failed"
    exit 1
fi

echo "Successfully completed PLS analysis for batch $SLURM_ARRAY_TASK_ID"