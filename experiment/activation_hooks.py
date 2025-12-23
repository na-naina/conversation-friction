"""Activation collection hooks for mechanistic interpretability.

Collects residual stream activations at specified layers during model inference.
Supports both raw activation collection and SAE encoding.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM


@dataclass
class ActivationCache:
    """Cache for storing activations during forward pass."""

    # Mapping: layer_idx -> list of activations (one per generation step)
    activations: dict[int, list[Tensor]] = field(default_factory=dict)

    # Track which token positions these correspond to
    token_positions: list[int] = field(default_factory=list)

    def clear(self) -> None:
        """Clear all cached activations."""
        self.activations.clear()
        self.token_positions.clear()

    def add_activation(self, layer_idx: int, activation: Tensor, position: int) -> None:
        """Add activation for a layer at a position."""
        if layer_idx not in self.activations:
            self.activations[layer_idx] = []
        self.activations[layer_idx].append(activation.detach().cpu())
        if position not in self.token_positions:
            self.token_positions.append(position)

    def get_activations(self, layer_idx: int) -> Tensor | None:
        """Get all activations for a layer, stacked.

        Returns:
            Tensor of shape (num_tokens, hidden_size) or None
        """
        if layer_idx not in self.activations:
            return None
        return torch.stack(self.activations[layer_idx], dim=0)


class ActivationCollector:
    """Collects activations from specified layers during generation.

    Uses forward hooks to capture residual stream activations at the output
    of specified transformer layers.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        layer_indices: list[int],
        num_tokens: int = 5,
    ):
        """Initialize activation collector.

        Args:
            model: The model to collect activations from
            layer_indices: Which layer indices to collect from
            num_tokens: How many tokens to collect at response start
        """
        self.model = model
        self.layer_indices = layer_indices
        self.num_tokens = num_tokens

        self.cache = ActivationCache()
        self._hooks: list[Any] = []
        self._tokens_collected = 0
        self._collecting = False

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific layer.

        Works with Gemma architecture where layers are in model.layers
        """
        # Try common architectures
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Gemma 3 1B (text-only): Gemma3ForCausalLM
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, "language_model"):
            # Gemma 3 4B+ (multimodal): Gemma3ForConditionalGeneration
            # Structure: model.language_model.model.layers
            lm = self.model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers[layer_idx]
            raise ValueError(
                f"Multimodal model but cannot find layers at language_model.model.layers"
            )
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            return self.model.transformer.h[layer_idx]
        else:
            raise ValueError(
                f"Unknown model architecture. Cannot find layer {layer_idx}"
            )

    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook for a specific layer."""
        def hook(
            module: nn.Module,
            input: tuple[Tensor, ...],
            output: tuple[Tensor, ...] | Tensor,
        ) -> None:
            if not self._collecting:
                return

            if self._tokens_collected >= self.num_tokens:
                return

            # Get the hidden states (first element if tuple)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Only take the last token's activation (most recently generated)
            # Shape: (batch=1, seq_len, hidden_size) -> (hidden_size,)
            last_token_activation = hidden_states[0, -1, :]

            self.cache.add_activation(
                layer_idx=layer_idx,
                activation=last_token_activation,
                position=self._tokens_collected,
            )

        return hook

    def start_collection(self) -> None:
        """Start collecting activations (register hooks)."""
        self.cache.clear()
        self._tokens_collected = 0
        self._collecting = True

        # Register hooks
        self._hooks = []
        for layer_idx in self.layer_indices:
            module = self._get_layer_module(layer_idx)
            hook = module.register_forward_hook(self._create_hook(layer_idx))
            self._hooks.append(hook)

    def stop_collection(self) -> None:
        """Stop collecting and remove hooks."""
        self._collecting = False
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def on_token_generated(self) -> None:
        """Call this after each token is generated."""
        if self._collecting:
            self._tokens_collected += 1

    def get_collected_activations(self) -> dict[int, Tensor]:
        """Get all collected activations.

        Returns:
            Dict mapping layer_idx -> tensor of shape (num_tokens, hidden_size)
        """
        return {
            layer_idx: self.cache.get_activations(layer_idx)
            for layer_idx in self.layer_indices
            if self.cache.get_activations(layer_idx) is not None
        }


@dataclass
class TurnActivations:
    """Activations collected for a single conversation turn."""

    turn_number: int
    conversation_id: str
    condition: str
    layer_activations: dict[int, Tensor]  # layer_idx -> (num_tokens, hidden_size)

    def save(self, path: Path) -> None:
        """Save activations to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "turn_number": self.turn_number,
                "conversation_id": self.conversation_id,
                "condition": self.condition,
                "layer_activations": self.layer_activations,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TurnActivations":
        """Load activations from disk."""
        data = torch.load(path, weights_only=True)
        return cls(
            turn_number=data["turn_number"],
            conversation_id=data["conversation_id"],
            condition=data["condition"],
            layer_activations=data["layer_activations"],
        )


class ActivationCollectorWithGeneration(ActivationCollector):
    """Extended collector that integrates with generation loop.

    Uses a custom generation callback to track token generation.
    """

    def generate_with_collection(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[str, dict[int, Tensor]]:
        """Generate response while collecting activations.

        Args:
            tokenizer: Tokenizer for the model
            messages: Chat messages
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Returns:
            Tuple of (response_text, layer_activations)
        """
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]

        # Start collection
        self.start_collection()

        try:
            # Generate token by token to track each generation
            generated_ids = inputs["input_ids"].clone()

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(generated_ids)
                    next_token_logits = outputs.logits[0, -1, :]

                    if temperature > 0:
                        probs = torch.softmax(next_token_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = next_token_logits.argmax().unsqueeze(0)

                    generated_ids = torch.cat(
                        [generated_ids, next_token.unsqueeze(0)],
                        dim=1,
                    )

                # Track token generation (for first N tokens only)
                self.on_token_generated()

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        finally:
            self.stop_collection()

        # Decode response
        new_tokens = generated_ids[0, input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip(), self.get_collected_activations()


def collect_conversation_activations(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    messages_by_turn: list[list[dict[str, str]]],
    layer_indices: list[int],
    conversation_id: str,
    condition: str,
    num_tokens: int = 5,
    output_dir: Path | None = None,
) -> list[TurnActivations]:
    """Collect activations for all turns in a conversation.

    Args:
        model: The model
        tokenizer: The tokenizer
        messages_by_turn: Messages for each turn (cumulative history)
        layer_indices: Which layers to collect from
        conversation_id: ID for this conversation
        condition: Experimental condition
        num_tokens: Tokens to collect per turn
        output_dir: Optional directory to save activations

    Returns:
        List of TurnActivations for each turn
    """
    collector = ActivationCollectorWithGeneration(
        model=model,
        layer_indices=layer_indices,
        num_tokens=num_tokens,
    )

    all_activations = []

    for turn_num, messages in enumerate(messages_by_turn, 1):
        # Generate and collect
        _, layer_acts = collector.generate_with_collection(
            tokenizer=tokenizer,
            messages=messages,
        )

        turn_acts = TurnActivations(
            turn_number=turn_num,
            conversation_id=conversation_id,
            condition=condition,
            layer_activations=layer_acts,
        )

        all_activations.append(turn_acts)

        # Optionally save
        if output_dir is not None:
            path = output_dir / conversation_id / f"turn_{turn_num:02d}.pt"
            turn_acts.save(path)

    return all_activations
