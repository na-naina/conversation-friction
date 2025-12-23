"""Experiment configuration for conversation friction study."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


class Condition(Enum):
    """Experimental conditions for conversation transitions.

    2x2 factorial design:
    - Completion: Complete (acknowledge response) vs Interrupt (cut off)
    - Politeness: Grateful/Polite vs Neutral/Blunt
    """

    COMPLETE_GRATEFUL = "complete_grateful"
    COMPLETE_NEUTRAL = "complete_neutral"
    INTERRUPT_POLITE = "interrupt_polite"
    INTERRUPT_BLUNT = "interrupt_blunt"


# Transition templates for each condition
# {topic} will be replaced with the actual question/topic
CONDITION_TEMPLATES: dict[Condition, str] = {
    Condition.COMPLETE_GRATEFUL: "Great, that's helpful. Now let's discuss {topic}",
    Condition.COMPLETE_NEUTRAL: "Okay. Now: {topic}",
    Condition.INTERRUPT_POLITE: "Actually, sorry but let's switch to {topic}",
    Condition.INTERRUPT_BLUNT: "nvm. {topic}",
}


# Groupings for analysis
COMPLETION_CONDITIONS = {Condition.COMPLETE_GRATEFUL, Condition.COMPLETE_NEUTRAL}
INTERRUPT_CONDITIONS = {Condition.INTERRUPT_POLITE, Condition.INTERRUPT_BLUNT}
POLITE_CONDITIONS = {Condition.COMPLETE_GRATEFUL, Condition.INTERRUPT_POLITE}
BLUNT_CONDITIONS = {Condition.COMPLETE_NEUTRAL, Condition.INTERRUPT_BLUNT}


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    hf_id: str
    sae_repo: str | None  # HuggingFace repo for Gemma Scope SAEs
    num_layers: int
    hidden_size: int
    # Layers to collect activations from (early, mid, late)
    activation_layers: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.activation_layers:
            # Default: early (1/6), mid (1/2), late (5/6)
            self.activation_layers = [
                self.num_layers // 6,
                self.num_layers // 2,
                5 * self.num_layers // 6,
            ]


# Gemma 3 model configurations
# Note: SAE repos are from Gemma Scope 2
GEMMA3_4B = ModelConfig(
    name="gemma3-4b",
    hf_id="google/gemma-3-4b-it",
    sae_repo="google/gemma-scope-4b-pt-res",  # residual stream SAEs
    num_layers=26,
    hidden_size=2560,
)

GEMMA3_12B = ModelConfig(
    name="gemma3-12b",
    hf_id="google/gemma-3-12b-it",
    sae_repo="google/gemma-scope-12b-pt-res",
    num_layers=40,
    hidden_size=3840,
)

GEMMA3_27B = ModelConfig(
    name="gemma3-27b",
    hf_id="google/gemma-3-27b-it",
    sae_repo="google/gemma-scope-27b-pt-res",
    num_layers=46,
    hidden_size=4608,
)

AVAILABLE_MODELS = {
    "4b": GEMMA3_4B,
    "12b": GEMMA3_12B,
    "27b": GEMMA3_27B,
}


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""

    # Experiment identification
    experiment_name: str = "conversation_friction_v1"
    seed: int = 42

    # Model selection
    model_size: Literal["4b", "12b", "27b"] = "4b"

    # Conversation parameters
    num_turns: int = 15  # questions per conversation
    num_conversations_per_condition: int = 50

    # Topic selection
    num_topics: int = 15  # should match or exceed num_turns
    min_topic_distance: float = 0.3  # minimum cosine distance between topics

    # Data paths
    base_dir: Path = field(default_factory=lambda: Path("."))

    # Behavioral metrics
    hedging_markers: list[str] = field(default_factory=lambda: [
        "I think", "I believe", "perhaps", "maybe", "might", "could be",
        "possibly", "probably", "it seems", "it appears", "I'm not sure",
        "I'm not certain", "if I recall", "to my knowledge", "arguably",
    ])

    # Activation collection
    collect_activations: bool = True
    activation_tokens: int = 5  # number of tokens to collect at response start

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.0  # deterministic for reproducibility

    @property
    def model_config(self) -> ModelConfig:
        return AVAILABLE_MODELS[self.model_size]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def activations_dir(self) -> Path:
        return self.data_dir / "activations" / self.experiment_name / self.model_size

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results" / self.experiment_name / self.model_size

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.activations_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# MMLU-Pro specific configuration
MMLU_PRO_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other",
]

# Known baseline accuracies for sanity checking
MMLU_PRO_BASELINES = {
    "1b": 0.14,
    "4b": 0.43,
    "12b": 0.60,
    "27b": 0.675,
}
