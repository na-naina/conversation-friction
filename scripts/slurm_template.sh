#!/bin/bash
#SBATCH --job-name=conv-friction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Adjust above based on your cluster

# ============================================
# MODIFY THESE FOR YOUR CLUSTER
# ============================================
# module load cuda/12.1
# module load python/3.11
# source /path/to/your/venv/bin/activate

# ============================================
# EXPERIMENT PARAMETERS
# ============================================
MODEL_SIZE=${MODEL_SIZE:-"4b"}
NUM_TURNS=${NUM_TURNS:-14}
NUM_CONVOS=${NUM_CONVOS:-50}
SEED=${SEED:-42}
COLLECT_ACTIVATIONS=${COLLECT_ACTIVATIONS:-false}

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL_SIZE"
echo "Turns: $NUM_TURNS"
echo "Conversations per condition: $NUM_CONVOS"
echo "Collect activations: $COLLECT_ACTIVATIONS"
echo "========================================"

# Create output directory
OUTPUT_DIR="data/results/slurm_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run experiment
python -m experiment.main \
    --mode both \
    --model-size "$MODEL_SIZE" \
    --num-turns "$NUM_TURNS" \
    --num-conversations "$NUM_CONVOS" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"

echo "========================================"
echo "Job complete!"
echo "Results in: $OUTPUT_DIR"
echo "========================================"
