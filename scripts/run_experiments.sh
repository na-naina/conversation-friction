#!/bin/bash
# Run full experiment suite
# Usage: ./scripts/run_experiments.sh [1b|4b|12b|all]

set -e

# Activate venv if not already
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
fi

MODEL_SIZE=${1:-"4b"}
NUM_TURNS=14  # Max 14 (number of MMLU-Pro categories)
NUM_CONVOS=50

run_experiment() {
    local size=$1
    echo ""
    echo "========================================"
    echo "Running $size experiment"
    echo "Turns: $NUM_TURNS, Conversations per condition: $NUM_CONVOS"
    echo "Total conversations: $((NUM_CONVOS * 4))"
    echo "========================================"
    echo ""

    python -m experiment.main \
        --mode both \
        --model-size "$size" \
        --num-turns "$NUM_TURNS" \
        --num-conversations "$NUM_CONVOS" \
        --seed 42

    echo ""
    echo "$size experiment complete!"
    echo "Results saved to: data/results/conversation_friction_v1/$size/"
}

case $MODEL_SIZE in
    "1b")
        run_experiment "1b"
        ;;
    "4b")
        run_experiment "4b"
        ;;
    "12b")
        echo "Warning: 12B requires ~24GB VRAM. May need quantization."
        run_experiment "12b"
        ;;
    "all")
        run_experiment "1b"
        run_experiment "4b"
        echo "Skipping 12B by default (tight on VRAM). Run with '12b' explicitly if needed."
        ;;
    "quick")
        # Quick test run
        echo "Running quick test (1b, 3 turns, 2 convos)..."
        python -m experiment.main \
            --mode both \
            --model-size 1b \
            --num-turns 3 \
            --num-conversations 2
        ;;
    *)
        echo "Usage: $0 [1b|4b|12b|all|quick]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All requested experiments complete!"
echo "========================================"
