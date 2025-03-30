#!/bin/bash
set -e  # Exit on error

# Define constants
PAIRS_PER_BATCH=25

# Generate configuration and pairs
python3 - <<EOF
import yaml
import json
from itertools import combinations
import os
import sys

try:
    # Read config
    with open('/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['subjects', 'movies', 'metric', 'alignment', 'roi_names', 'data_path']
    missing = [field for field in required_fields if field not in config]
    if missing:
        print(f"Error: Missing required fields in config.yaml: {missing}")
        sys.exit(1)

    # Generate all subject pairs
    pairs = list(combinations(config['subjects'], 2))
    total_pairs = len(pairs)

    # Write pairs to file
    with open('/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/subject_pairs.txt', 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]},{pair[1]}\n")

    # Calculate and write batch ranges
    num_batches = (total_pairs + ${PAIRS_PER_BATCH} - 1) // ${PAIRS_PER_BATCH}
    with open('/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/batch_ranges.txt', 'w') as f:
        for i in range(num_batches):
            start_idx = i * ${PAIRS_PER_BATCH}
            end_idx = min((i + 1) * ${PAIRS_PER_BATCH} - 1, total_pairs - 1)
            f.write(f"{start_idx},{end_idx}\n")

    # Export config parameters
    params = {
        'TOTAL_PAIRS': total_pairs,
        'NUM_BATCHES': num_batches,
        'SUBJECTS': ','.join(config['subjects']),
        'MOVIES': ','.join(config['movies']),
        'METRIC': config['metric'],
        'ALIGNMENT': config['alignment'],
        'PERFORM_PERMUTATIONS': str(config.get('perform_permutations', False)),
        'N_PERMUTATIONS': str(config.get('n_permutations', 0)),
        'PERFORM_DOWNSAMPLING': str(config.get('perform_downsampling', False)),
        'N_DOWNSAMPLE': str(config.get('n_downsample', 0)),
        'RANDOM_STATE': str(config.get('random_state', 42)),
        'BLOCK_SIZE': str(config.get('block_size', 10)),
        'DATA_PATH': config['data_path'],
        'ROI_NAMES': ';'.join(config['roi_names'])  # Use semicolon to avoid space issues
    }

    with open('/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/config_params.sh', 'w') as f:
        for key, value in params.items():
            f.write(f'export CONFIG_{key}="{value}"\n')

    print(f"Configuration processed successfully:")
    print(f"Total pairs: {total_pairs}")
    print(f"Number of batches: {num_batches}")
    print(f"Batch size: ${PAIRS_PER_BATCH}")

except Exception as e:
    print(f"Error processing configuration: {str(e)}")
    sys.exit(1)
EOF

# Check if required files were created
for file in "/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/subject_pairs.txt" "/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/batch_ranges.txt" "/scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/config_params.sh"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file was not created"
        exit 1
    fi
done

# Source the config parameters
source /scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/config_params.sh

# Validate required environment variables
required_vars=("CONFIG_SUBJECTS" "CONFIG_MOVIES" "CONFIG_METRIC" "CONFIG_ALIGNMENT" "CONFIG_ROI_NAMES")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# Create log directory
if [ "$CONFIG_PERFORM_PERMUTATIONS" = "True" ]; then
    log_dir="log/${CONFIG_METRIC}_${CONFIG_ALIGNMENT}_perm"
else
    log_dir="log/${CONFIG_METRIC}_${CONFIG_ALIGNMENT}"
fi
mkdir -p "$log_dir"

# Submit jobs for each ROI
IFS=';' read -ra ROIS <<< "$CONFIG_ROI_NAMES"
for roi in "${ROIS[@]}"; do
    echo "Submitting jobs for ROI: $roi"
    
    job_id=$(sbatch --array=0-$((CONFIG_NUM_BATCHES - 1)) \
                    --parsable \
                    --export=ALL,ROI="$roi" \
                    --output="${log_dir}/${roi}_batch%a_%A.out" \
                    --error="${log_dir}/${roi}_batch%a_%A.err" \
                    /scratch4/mbonner5/chan21/indiv_dim/slurm_jobs/slurm_script.sh)
    
    if [ $? -ne 0 ]; then
        echo "Error submitting jobs for ROI: $roi"
        exit 1
    fi
    
    echo "Submitted job array $job_id for ROI: $roi"
done