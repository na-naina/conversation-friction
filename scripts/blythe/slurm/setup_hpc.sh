#!/bin/bash
# One-time setup on Blythe HPC (run this interactively, NOT via sbatch)
# SSH to blythe first, then run this script

set -e

echo "Setting up conversation-friction on Blythe HPC..."

# Create project directory
mkdir -p $SHARE/u5584851/conversation-friction
cd $SHARE/u5584851/conversation-friction

# Create subdirectories
mkdir -p logs data/results data/activations

# Setup Python venv
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies (will happen on first job run too)
# pip install -e .  # Uncomment after rsync

# HuggingFace login
echo ""
echo "========================================"
echo "HuggingFace Authentication"
echo "========================================"
echo "Run: huggingface-cli login"
echo "Paste your token from: https://huggingface.co/settings/tokens"
echo ""
echo "Also accept Gemma license at:"
echo "  https://huggingface.co/google/gemma-3-4b-it"
echo "  https://huggingface.co/google/gemma-3-12b-it"
echo "  https://huggingface.co/google/gemma-3-27b-it"
echo "========================================"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. From local machine: ./scripts/blythe/sync_to_hpc.sh"
echo "2. On HPC: huggingface-cli login"
echo "3. On HPC: cd \$SHARE/u5584851/conversation-friction && sbatch slurm/friction_4b.slurm"
