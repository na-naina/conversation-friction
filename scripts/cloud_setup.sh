#!/bin/bash
# Cloud instance setup script for conversation-friction experiments
# Tested on: Ubuntu 22.04 with RTX 4090

set -e  # Exit on error

echo "================================================"
echo "Setting up conversation-friction experiment"
echo "================================================"

# 1. System updates (optional, skip if fresh instance)
# sudo apt update && sudo apt install -y git python3-pip python3-venv

# 2. Create working directory
mkdir -p ~/experiments
cd ~/experiments

# 3. Clone and install
echo "Cloning repository..."
git clone https://github.com/na-naina/conversation-friction.git
cd conversation-friction

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .

# 4. HuggingFace authentication
echo ""
echo "================================================"
echo "HuggingFace Authentication Required"
echo "================================================"
echo "You need to authenticate to download Gemma models."
echo ""
echo "Option 1: Run 'huggingface-cli login' and paste your token"
echo "Option 2: Export HF_TOKEN=your_token_here"
echo ""
echo "Get your token at: https://huggingface.co/settings/tokens"
echo "Make sure you've accepted the Gemma license at:"
echo "  https://huggingface.co/google/gemma-3-1b-it"
echo "================================================"

# 5. Create data directories
mkdir -p data/activations data/results

# 6. Quick sanity check
echo ""
echo "Testing Python environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Authenticate with HuggingFace: huggingface-cli login"
echo "2. Run experiments (see below)"
echo ""
echo "Quick test (1B, 2 conversations, 3 turns):"
echo "  python -m experiment.main --model-size 1b --num-turns 3 --num-conversations 2"
echo ""
echo "Full 1B experiment:"
echo "  python -m experiment.main --model-size 1b --num-turns 15 --num-conversations 50"
echo ""
echo "Full 4B experiment:"
echo "  python -m experiment.main --model-size 4b --num-turns 15 --num-conversations 50"
echo ""
