#!/bin/bash
#SBATCH --job-name=probe_12b
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Clean Interruption Probe Experiment: Gemma 3 12B
# This isolates the "was I interrupted?" signal for interpretability

set -e
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | Start: $(date)"

cd $SHARE/u5584851/conversation-friction
module load GCC/12.3.0 && module load CUDA/12.1.1 && module load Python/3.11.3-GCCcore-12.3.0

export HF_HOME=$SHARE/u5584851/hf_cache
export TRANSFORMERS_CACHE=$SHARE/u5584851/hf_cache
export HF_DATASETS_CACHE=$SHARE/u5584851/hf_cache/datasets

mkdir -p logs data/probe_experiment

if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -e .
else
    source venv/bin/activate
fi

echo "Starting clean interruption probe experiment (12B)..."
python -m experiment.interruption_probe \
    --model-size 12b \
    --num-repeats 20 \
    --truncate-tokens 50

echo "Complete: $(date)"
