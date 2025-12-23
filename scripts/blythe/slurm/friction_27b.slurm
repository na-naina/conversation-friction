#!/bin/bash
#SBATCH --job-name=friction_27b
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Conversation Friction Experiment: Gemma 3 27B (requires 2 GPUs)

set -e
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

cd $SHARE/u5584851/conversation-friction
module load GCC/12.3.0 && module load CUDA/12.1.1

export HF_HOME=$SHARE/u5584851/hf_cache
export TRANSFORMERS_CACHE=$SHARE/u5584851/hf_cache
export HF_DATASETS_CACHE=$SHARE/u5584851/hf_cache/datasets

mkdir -p logs data/results data/activations

if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    source venv/bin/activate
fi

echo "Starting 27B experiment with activation collection..."
python -m experiment.main \
    --mode both \
    --model-size 27b \
    --num-turns 14 \
    --num-conversations 25 \
    --seed 42 \
    --collect-activations

echo "Complete: $(date)"
