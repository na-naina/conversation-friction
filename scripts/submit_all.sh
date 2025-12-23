#!/bin/bash
# Submit jobs for all model sizes
# Usage: ./scripts/submit_all.sh

set -e

echo "Submitting conversation friction experiments..."

# 4B - Primary experiment
echo "Submitting 4B job..."
sbatch --export=MODEL_SIZE=4b,NUM_CONVOS=50 scripts/slurm_template.sh

# 12B - Scaling validation
echo "Submitting 12B job..."
sbatch --export=MODEL_SIZE=12b,NUM_CONVOS=50 scripts/slurm_template.sh

# 27B - Full scale (needs more memory/time)
echo "Submitting 27B job..."
sbatch --export=MODEL_SIZE=27b,NUM_CONVOS=25 \
       --mem=128G --time=24:00:00 \
       scripts/slurm_template.sh

echo "All jobs submitted! Check with: squeue -u \$USER"
