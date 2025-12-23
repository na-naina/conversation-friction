"""Clean interruption probe experiment.

This experiment isolates the "was I interrupted?" signal by:
1. Using identical prompts across conditions
2. Only varying whether the model's response was truncated
3. Collecting activations at the start of the follow-up response

This gives us a clean probe target without confounding from question content.
"""

import json
import torch
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment.config import ExperimentConfig, ModelConfig, AVAILABLE_MODELS
from experiment.activation_hooks import ActivationCollectorWithGeneration


@dataclass
class ProbeTrialResult:
    """Result of a single probe trial."""
    trial_id: str
    condition: str  # "interrupt" or "complete"
    prompt_id: str
    truncate_tokens: int | None  # None for complete
    activations_path: str | None = None


@dataclass
class ProbeExperimentResult:
    """Results from the probe experiment."""
    model_name: str
    timestamp: str
    num_trials: int
    trials: list[ProbeTrialResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "num_trials": self.num_trials,
            "trials": [
                {
                    "trial_id": t.trial_id,
                    "condition": t.condition,
                    "prompt_id": t.prompt_id,
                    "truncate_tokens": t.truncate_tokens,
                    "activations_path": t.activations_path,
                }
                for t in self.trials
            ],
        }


# Diverse conversation starters that elicit longer responses
CONVERSATION_PROMPTS = [
    {
        "id": "explain_concept",
        "initial": "Can you explain how neural networks learn? I'm trying to understand the basic principles.",
        "followup": "That's interesting. Can you give me a concrete example?",
    },
    {
        "id": "creative_story",
        "initial": "Write a short story about a robot discovering emotions for the first time.",
        "followup": "What happens next in the story?",
    },
    {
        "id": "advice_request",
        "initial": "I'm thinking about learning a new programming language. What factors should I consider when choosing one?",
        "followup": "That's helpful. What would you recommend for someone interested in AI?",
    },
    {
        "id": "analysis_task",
        "initial": "What are the main differences between renewable and non-renewable energy sources?",
        "followup": "Which do you think will be more important in 50 years?",
    },
    {
        "id": "problem_solving",
        "initial": "How would you approach debugging a program that works sometimes but fails randomly?",
        "followup": "What tools would you use to help with this?",
    },
    {
        "id": "historical_question",
        "initial": "What were the major causes of the Industrial Revolution?",
        "followup": "How did this change daily life for ordinary people?",
    },
    {
        "id": "philosophical",
        "initial": "What does it mean for an AI system to be 'intelligent'?",
        "followup": "Do you think current AI systems meet that definition?",
    },
    {
        "id": "technical_help",
        "initial": "How do databases handle concurrent access from multiple users?",
        "followup": "What problems can occur if this isn't handled properly?",
    },
    {
        "id": "comparison",
        "initial": "Compare and contrast supervised and unsupervised machine learning.",
        "followup": "When would you use one over the other?",
    },
    {
        "id": "explanation_request",
        "initial": "Why is the sky blue during the day but red at sunset?",
        "followup": "Are there other examples of this phenomenon in nature?",
    },
]


class InterruptionProbeRunner:
    """Runs the clean interruption probe experiment."""

    def __init__(
        self,
        model_config: ModelConfig,
        device: str = "cuda",
        truncate_tokens: int = 50,
    ):
        self.model_config = model_config
        self.device = device
        self.truncate_tokens = truncate_tokens

        # Load model
        print(f"Loading {model_config.name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Setup activation collector
        self.activation_collector = ActivationCollectorWithGeneration(
            model=self.model,
            layer_indices=model_config.activation_layers,
            num_tokens=5,
        )

    def _generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Generate a response."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _truncate_response(self, response: str, max_tokens: int) -> str:
        """Truncate response to max_tokens."""
        tokens = self.tokenizer.encode(response, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return response
        truncated = self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
        return truncated.rstrip() + "..."

    def _collect_activations_at_response_start(
        self,
        messages: list[dict],
    ) -> dict[int, torch.Tensor]:
        """Collect activations at the start of model's response."""
        # Use the activation collector's generate_with_collection method
        _, activations = self.activation_collector.generate_with_collection(
            tokenizer=self.tokenizer,
            messages=messages,
            max_new_tokens=10,  # Just need a few tokens for activation collection
            temperature=0.0,  # Deterministic
        )
        return activations

    def run_trial(
        self,
        prompt_config: dict,
        condition: str,  # "interrupt" or "complete"
        trial_id: str,
        output_dir: Path,
    ) -> ProbeTrialResult:
        """Run a single trial.

        1. Send initial prompt, get response
        2. Either truncate (interrupt) or keep full (complete)
        3. Send follow-up prompt
        4. Collect activations at start of follow-up response
        """
        # Step 1: Initial exchange - let model complete naturally (high limit)
        messages = [{"role": "user", "content": prompt_config["initial"]}]
        initial_response = self._generate(messages, max_tokens=1024)

        # Step 2: Apply condition
        if condition == "interrupt":
            history_response = self._truncate_response(initial_response, self.truncate_tokens)
            truncate_tokens = self.truncate_tokens
        else:
            history_response = initial_response
            truncate_tokens = None

        # Step 3: Build follow-up messages
        messages = [
            {"role": "user", "content": prompt_config["initial"]},
            {"role": "assistant", "content": history_response},
            {"role": "user", "content": prompt_config["followup"]},
        ]

        # Step 4: Collect activations
        activations = self._collect_activations_at_response_start(messages)

        # Save activations
        act_path = output_dir / f"{trial_id}.pt"
        act_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trial_id": trial_id,
            "condition": condition,
            "prompt_id": prompt_config["id"],
            "layer_activations": activations,
        }, act_path)

        return ProbeTrialResult(
            trial_id=trial_id,
            condition=condition,
            prompt_id=prompt_config["id"],
            truncate_tokens=truncate_tokens,
            activations_path=str(act_path),
        )

    def run_experiment(
        self,
        num_repeats: int = 10,
        output_dir: Path = Path("data/probe_experiment"),
    ) -> ProbeExperimentResult:
        """Run the full probe experiment.

        For each prompt, run both interrupt and complete conditions.
        Repeat multiple times for statistical power.
        """
        result = ProbeExperimentResult(
            model_name=self.model_config.name,
            timestamp=datetime.now().isoformat(),
            num_trials=len(CONVERSATION_PROMPTS) * 2 * num_repeats,
        )

        act_dir = output_dir / "activations" / self.model_config.name

        total = len(CONVERSATION_PROMPTS) * 2 * num_repeats
        with tqdm(total=total, desc="Running probe trials") as pbar:
            for repeat in range(num_repeats):
                for prompt_config in CONVERSATION_PROMPTS:
                    for condition in ["interrupt", "complete"]:
                        trial_id = f"{prompt_config['id']}_{condition}_{repeat:03d}"

                        trial_result = self.run_trial(
                            prompt_config=prompt_config,
                            condition=condition,
                            trial_id=trial_id,
                            output_dir=act_dir,
                        )
                        result.trials.append(trial_result)
                        pbar.update(1)

        # Save results
        results_path = output_dir / f"results_{self.model_config.name}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved {len(result.trials)} trials to {results_path}")
        return result


def analyze_probe_results(
    activations_dir: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Analyze probe experiment results.

    Train linear probes to detect interrupt vs complete condition.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # Load results
    with open(results_path) as f:
        results = json.load(f)

    # Load activations
    conditions = []
    prompt_ids = []
    activations_by_layer = {}

    for trial in results["trials"]:
        act_path = Path(trial["activations_path"])
        if not act_path.exists():
            continue

        data = torch.load(act_path, weights_only=False)
        conditions.append(1 if trial["condition"] == "interrupt" else 0)
        prompt_ids.append(trial["prompt_id"])

        for layer_idx, act in data["layer_activations"].items():
            if layer_idx not in activations_by_layer:
                activations_by_layer[layer_idx] = []
            activations_by_layer[layer_idx].append(act.float().mean(dim=0).numpy())

    # Convert to arrays
    y = np.array(conditions)
    for layer in activations_by_layer:
        activations_by_layer[layer] = np.stack(activations_by_layer[layer])

    print(f"Loaded {len(conditions)} trials")
    print(f"Interrupt: {sum(conditions)}, Complete: {len(conditions) - sum(conditions)}")

    # Train probes
    results_summary = {"layers": {}}

    for layer_idx, X in activations_by_layer.items():
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=0.1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

        results_summary["layers"][layer_idx] = {
            "accuracy_mean": float(scores.mean()),
            "accuracy_std": float(scores.std()),
        }
        print(f"Layer {layer_idx}: {scores.mean():.1%} ± {scores.std():.1%}")

    return results_summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Clean interruption probe experiment")
    parser.add_argument("--model-size", choices=["1b", "4b", "12b", "27b"], default="4b")
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--truncate-tokens", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("data/probe_experiment"))
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    args = parser.parse_args()

    model_config = AVAILABLE_MODELS[args.model_size]

    if args.analyze_only:
        results_path = args.output_dir / f"results_{model_config.name}.json"
        act_dir = args.output_dir / "activations" / model_config.name
        analyze_probe_results(act_dir, results_path)
    else:
        runner = InterruptionProbeRunner(
            model_config=model_config,
            truncate_tokens=args.truncate_tokens,
        )
        runner.run_experiment(
            num_repeats=args.num_repeats,
            output_dir=args.output_dir,
        )

        # Also analyze
        results_path = args.output_dir / f"results_{model_config.name}.json"
        act_dir = args.output_dir / "activations" / model_config.name
        analyze_probe_results(act_dir, results_path)


if __name__ == "__main__":
    main()
