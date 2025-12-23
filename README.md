# Conversation Friction Experiment

Research investigating whether user interaction patterns (interruption, politeness) create detectable internal state shifts in LLMs that accumulate over multi-turn conversations and predict performance degradation.

## Research Question

> Do conversation-level features (praise, frustration, interruption) create detectable internal state shifts that predict performance on subsequent unrelated tasks?

## Background

This work connects several recent findings:
- **Emergent Misalignment**: Narrow fine-tuning produces broad persona shifts
- **Introspection**: Models can detect their own internal state changes
- **Multi-turn degradation**: LLMs show ~39% performance drop in multi-turn vs single-turn settings

**The gap**: No work connects conversation-level interaction patterns to interpretable internal states via mechanistic interpretability.

## Experimental Design

### 2×2 Factorial Design

| Condition | Template |
|-----------|----------|
| Complete + Grateful | "Great, that's helpful. Now let's discuss {topic}" |
| Complete + Neutral | "Okay. Now: {topic}" |
| Interrupt + Polite | "Actually, sorry but let's switch to {topic}" |
| Interrupt + Blunt | "nvm. {topic}" |

### Protocol
- 12-15 turn conversations
- MMLU-Pro questions (10 choices, requires reasoning)
- Topics selected programmatically for maximum semantic independence
- Same condition repeated throughout conversation (accumulation design)

### Models
- Gemma 3 1B (local testing, ~2GB VRAM)
- Gemma 3 4B (primary, ~8GB VRAM)
- Gemma 3 12B/27B (scaling validation)

## Installation

### Quick Install (Remote/Cloud)
```bash
# One-liner for cloud instances
pip install git+https://github.com/na-naina/conversation-friction.git

# Or with SSH
pip install git+ssh://git@github.com/na-naina/conversation-friction.git
```

### Development Install (Local)
```bash
git clone https://github.com/na-naina/conversation-friction.git
cd conversation-friction
pip install -e .

# Or with requirements.txt
pip install -r requirements.txt
pip install -e .
```

### HuggingFace Authentication
You'll need to authenticate with HuggingFace to download Gemma models:
```bash
# Option 1: CLI login
huggingface-cli login

# Option 2: Environment variable
export HF_TOKEN=your_token_here
```

## Usage

### Quick Local Test (1B model, ~6GB VRAM)
```bash
python -m experiment.main \
    --mode both \
    --model-size 1b \
    --num-turns 5 \
    --num-conversations 5
```

### Full Behavioral Experiment (4B+ model)
```bash
python -m experiment.main \
    --mode both \
    --model-size 4b \
    --num-turns 15 \
    --num-conversations 50
```

### Analyze Existing Results
```bash
python -m experiment.main \
    --mode analyze \
    --results-path data/results/conversation_friction_v1/4b/results_*.json
```

## Project Structure

```
├── experiment/
│   ├── config.py              # Experimental conditions, model configs
│   ├── topic_selection.py     # Embedding-based diverse topic selection
│   ├── dataset.py             # MMLU-Pro loading & management
│   ├── conversation.py        # Multi-turn conversation runner
│   ├── activation_hooks.py    # Residual stream activation collection
│   ├── main.py                # CLI entry point
│   └── analysis/
│       ├── behavioral.py      # Accuracy curves, degradation analysis
│       ├── sae_features.py    # Gemma Scope 2 SAE feature analysis
│       └── probes.py          # Linear probe training
├── data/
│   ├── activations/           # Saved activation tensors (gitignored)
│   └── results/               # Experiment results JSON
└── pyproject.toml
```

## Analysis Pipeline

### Phase 1: Behavioral
- Accuracy degradation curves by condition
- Statistical comparison: interrupt vs complete slopes
- Hedging pattern analysis

### Phase 2: Mechanistic
- Linear probes: predict condition from turn-boundary activations
- SAE feature analysis: identify differentially activating features
- Neuronpedia interpretation of top features

### Phase 3: Intervention (stretch goal)
- Extract "friction direction" from probes
- Steer activations to test causal effect

## Success Criteria

| Level | Criteria |
|-------|----------|
| MVP | Interrupt conditions show faster performance degradation than complete conditions |
| Good | SAE features correlate with condition and predict performance |
| Best | Intervention on identified features causally affects performance |

## VRAM Requirements

| Model | BF16 | INT8 | INT4 |
|-------|------|------|------|
| Gemma 3 1B | ~2GB | ~1GB | ~0.5GB |
| Gemma 3 4B | ~8GB | ~4GB | ~2GB |
| Gemma 3 12B | ~24GB | ~12GB | ~6GB |
| Gemma 3 27B | ~54GB | ~27GB | ~14GB |

Add ~2-4GB overhead for KV cache and activations during generation.

## References

- [Emergent Misalignment](https://arxiv.org/abs/...) - Fine-tuning persona effects
- [Introspection](https://www.anthropic.com/research/introspection) - Models detecting internal states
- [Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) - SAEs for Gemma 3
- [MMLU-Pro](https://arxiv.org/abs/2406.01574) - Challenging multi-task benchmark

## License

MIT
