"""SAE feature analysis for conversation friction experiments.

Loads Gemma Scope 2 SAEs and analyzes which features activate differentially
across experimental conditions.

Based on the official Gemma Scope 2 tutorial:
https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import pandas as pd
from tqdm import tqdm
from scipy import stats


def get_sae_repo_id(model_size: str, instruction_tuned: bool = False) -> str:
    """Get the HuggingFace repo ID for Gemma Scope 2 SAEs.

    Args:
        model_size: Model size (e.g., "1b", "4b", "12b", "27b")
        instruction_tuned: Whether to use IT model SAEs (less tested)

    Returns:
        HuggingFace repo ID
    """
    suffix = "it" if instruction_tuned else "pt"
    return f"google/gemma-scope-2-{model_size}-{suffix}"


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder matching Gemma Scope 2 architecture.

    Based on the official implementation from the Gemma Scope 2 tutorial.
    Uses JumpReLU activation: mask * ReLU(pre_acts) where mask = (pre_acts > threshold)
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        # Initialize to zeros - we'll load pretrained weights
        self.w_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.w_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts: Tensor) -> Tensor:
        """Encode activations to sparse feature space.

        Args:
            input_acts: Tensor of shape (..., d_model)

        Returns:
            Feature activations of shape (..., d_sae)
        """
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: Tensor) -> Tensor:
        """Decode from feature space back to activation space.

        Args:
            acts: Feature activations of shape (..., d_sae)

        Returns:
            Reconstructed activations of shape (..., d_model)
        """
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass: encode then decode."""
        acts = self.encode(x)
        return self.decode(acts)


def load_gemma_scope_sae(
    model_size: str,
    layer: int,
    width: str = "16k",
    l0: str = "medium",
    category: str = "resid_post",
    instruction_tuned: bool = False,
    device: str = "cuda",
) -> JumpReLUSAE:
    """Load a Gemma Scope 2 SAE from HuggingFace.

    Args:
        model_size: Model size ("1b", "4b", "12b", "27b")
        layer: Layer index
        width: SAE width ("16k", "65k", "262k", "1m")
        l0: Sparsity level ("small", "medium", "big")
        category: Activation site ("resid_post", "mlp_out", "attn_out")
        instruction_tuned: Use IT model SAEs
        device: Device to load to

    Returns:
        Loaded JumpReLUSAE
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    repo_id = get_sae_repo_id(model_size, instruction_tuned)
    filename = f"{category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"

    print(f"Loading SAE from {repo_id}/{filename}")

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    params = load_file(path_to_params, device=device)

    d_model, d_sae = params["w_enc"].shape
    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(params)
    sae.to(device)

    return sae


# Keep the old API for backwards compatibility
class GemmaScopeSAE:
    """Wrapper for Gemma Scope SAE loading and encoding.

    This is a compatibility wrapper around JumpReLUSAE.
    """

    def __init__(
        self,
        repo_id: str,
        layer: int,
        width: str = "16k",
        l0: str = "medium",
        device: str = "cuda",
    ):
        self.repo_id = repo_id
        self.layer = layer
        self.width = width
        self.l0 = l0
        self.device = device

        # Parse model size from repo_id
        # e.g., "google/gemma-scope-2-1b-pt" -> "1b"
        parts = repo_id.split("-")
        if "gemma-scope-2" in repo_id:
            self.model_size = parts[-2]  # "1b", "4b", etc.
            self.instruction_tuned = parts[-1] == "it"
        else:
            # Fallback for old-style repo names
            self.model_size = "4b"
            self.instruction_tuned = False

        self._sae = load_gemma_scope_sae(
            model_size=self.model_size,
            layer=layer,
            width=width,
            l0=l0,
            instruction_tuned=self.instruction_tuned,
            device=device,
        )

        self.d_model = self._sae.w_enc.shape[0]
        self.n_features = self._sae.w_enc.shape[1]

    def encode(self, activations: Tensor) -> Tensor:
        """Encode activations to SAE feature space."""
        return self._sae.encode(activations.to(torch.float32))

    def decode(self, feature_acts: Tensor) -> Tensor:
        """Decode from feature space."""
        return self._sae.decode(feature_acts)


def load_sae_for_layer(
    model_size: str,
    layer: int,
    width: str = "16k",
    l0: str = "medium",
    device: str = "cuda",
) -> GemmaScopeSAE:
    """Load SAE for a specific layer.

    Args:
        model_size: Model size ("1b", "4b", "12b", "27b")
        layer: Layer index
        width: SAE width
        l0: Sparsity level
        device: Device

    Returns:
        Loaded SAE wrapper
    """
    repo_id = get_sae_repo_id(model_size)
    return GemmaScopeSAE(
        repo_id=repo_id,
        layer=layer,
        width=width,
        l0=l0,
        device=device,
    )


@dataclass
class FeatureActivationStats:
    """Statistics for a single SAE feature across conditions."""

    feature_idx: int
    layer: int

    # Mean activation by condition
    mean_by_condition: dict[str, float]

    # Overall statistics
    overall_mean: float
    overall_std: float

    # Differential activation (interrupt - complete)
    interrupt_complete_diff: float
    interrupt_complete_pvalue: float

    # Differential activation (polite - blunt)
    polite_blunt_diff: float
    polite_blunt_pvalue: float


def compute_feature_statistics(
    feature_acts_by_condition: dict[str, Tensor],
    layer: int,
    top_k: int = 100,
) -> list[FeatureActivationStats]:
    """Compute statistics for features with largest condition differences.

    Args:
        feature_acts_by_condition: Dict mapping condition -> (n_samples, n_features)
        layer: Layer index (for labeling)
        top_k: Number of top differential features to return

    Returns:
        List of FeatureActivationStats for top features
    """
    conditions = list(feature_acts_by_condition.keys())
    n_features = feature_acts_by_condition[conditions[0]].shape[1]

    # Identify interrupt vs complete conditions
    interrupt_conds = [c for c in conditions if "interrupt" in c]
    complete_conds = [c for c in conditions if "complete" in c]

    # Identify polite vs blunt conditions
    polite_conds = [c for c in conditions if "grateful" in c or "polite" in c]
    blunt_conds = [c for c in conditions if "neutral" in c or "blunt" in c]

    # Compute differences
    all_stats = []

    for feat_idx in range(n_features):
        # Get activations for this feature
        feat_by_cond = {
            c: feature_acts_by_condition[c][:, feat_idx].numpy()
            for c in conditions
        }

        # Mean by condition
        mean_by_cond = {c: float(np.mean(v)) for c, v in feat_by_cond.items()}
        overall = np.concatenate(list(feat_by_cond.values()))

        # Interrupt vs complete comparison
        interrupt_vals = np.concatenate([feat_by_cond[c] for c in interrupt_conds])
        complete_vals = np.concatenate([feat_by_cond[c] for c in complete_conds])
        ic_diff = np.mean(interrupt_vals) - np.mean(complete_vals)
        _, ic_pval = stats.ttest_ind(interrupt_vals, complete_vals)

        # Polite vs blunt comparison
        polite_vals = np.concatenate([feat_by_cond[c] for c in polite_conds])
        blunt_vals = np.concatenate([feat_by_cond[c] for c in blunt_conds])
        pb_diff = np.mean(polite_vals) - np.mean(blunt_vals)
        _, pb_pval = stats.ttest_ind(polite_vals, blunt_vals)

        all_stats.append(FeatureActivationStats(
            feature_idx=feat_idx,
            layer=layer,
            mean_by_condition=mean_by_cond,
            overall_mean=float(np.mean(overall)),
            overall_std=float(np.std(overall)),
            interrupt_complete_diff=float(ic_diff),
            interrupt_complete_pvalue=float(ic_pval),
            polite_blunt_diff=float(pb_diff),
            polite_blunt_pvalue=float(pb_pval),
        ))

    # Sort by absolute interrupt-complete difference
    all_stats.sort(key=lambda x: abs(x.interrupt_complete_diff), reverse=True)

    return all_stats[:top_k]


def encode_activations_with_sae(
    activations_dir: Path,
    sae: GemmaScopeSAE,
    layer: int,
) -> dict[str, Tensor]:
    """Encode all saved activations through SAE.

    Args:
        activations_dir: Directory with saved TurnActivations
        sae: SAE to use for encoding
        layer: Which layer's activations to encode

    Returns:
        Dict mapping condition -> (n_samples, n_features)
    """
    from experiment.activation_hooks import TurnActivations

    feature_acts_by_condition: dict[str, list[Tensor]] = {}

    # Find all activation files
    for conv_dir in activations_dir.iterdir():
        if not conv_dir.is_dir():
            continue

        for turn_file in conv_dir.glob("turn_*.pt"):
            turn_acts = TurnActivations.load(turn_file)

            if layer not in turn_acts.layer_activations:
                continue

            condition = turn_acts.condition
            if condition not in feature_acts_by_condition:
                feature_acts_by_condition[condition] = []

            # Get activation and encode
            raw_acts = turn_acts.layer_activations[layer]  # (n_tokens, hidden)
            # Skip first token if it might be BOS (SAEs aren't trained on BOS)
            if raw_acts.shape[0] > 1:
                raw_acts = raw_acts[1:]  # Skip potential BOS
            # Average over tokens
            mean_act = raw_acts.mean(dim=0).to(sae.device)
            feature_act = sae.encode(mean_act)

            feature_acts_by_condition[condition].append(feature_act.cpu())

    # Stack
    return {
        cond: torch.stack(acts)
        for cond, acts in feature_acts_by_condition.items()
    }


def find_differential_features(
    activations_dir: Path,
    model_size: str,
    layers: list[int],
    width: str = "16k",
    l0: str = "medium",
    top_k: int = 50,
    device: str = "cuda",
) -> dict[int, list[FeatureActivationStats]]:
    """Find features that activate differentially across conditions.

    Args:
        activations_dir: Directory with saved activations
        model_size: Model size for SAE loading
        layers: Which layers to analyze
        width: SAE width
        l0: Sparsity level
        top_k: Number of top features per layer
        device: Device for SAE

    Returns:
        Dict mapping layer -> list of top differential features
    """
    results = {}

    for layer in tqdm(layers, desc="Analyzing layers"):
        # Load SAE
        sae = load_sae_for_layer(model_size, layer, width, l0, device)

        # Encode activations
        feature_acts = encode_activations_with_sae(activations_dir, sae, layer)

        # Compute statistics
        layer_stats = compute_feature_statistics(feature_acts, layer, top_k)
        results[layer] = layer_stats

    return results


def features_to_dataframe(
    feature_stats: dict[int, list[FeatureActivationStats]],
) -> pd.DataFrame:
    """Convert feature statistics to DataFrame."""
    rows = []
    for layer, stats_list in feature_stats.items():
        for stat in stats_list:
            row = {
                "layer": layer,
                "feature_idx": stat.feature_idx,
                "overall_mean": stat.overall_mean,
                "overall_std": stat.overall_std,
                "interrupt_complete_diff": stat.interrupt_complete_diff,
                "interrupt_complete_pvalue": stat.interrupt_complete_pvalue,
                "polite_blunt_diff": stat.polite_blunt_diff,
                "polite_blunt_pvalue": stat.polite_blunt_pvalue,
            }
            for cond, mean in stat.mean_by_condition.items():
                row[f"mean_{cond}"] = mean
            rows.append(row)

    return pd.DataFrame(rows)


def get_neuronpedia_url(
    layer: int,
    feature_idx: int,
    model_size: str = "1b",
    width: str = "16k",
    l0: str = "medium",
) -> str:
    """Generate Neuronpedia URL for a feature.

    Note: Neuronpedia URL format may vary - check their site for current format.
    """
    # This is a guess at the format - Neuronpedia may use different conventions
    return (
        f"https://neuronpedia.org/gemma-scope-2-{model_size}-pt/"
        f"layer_{layer}_width_{width}_l0_{l0}/{feature_idx}"
    )


def create_feature_report(
    feature_stats: dict[int, list[FeatureActivationStats]],
    output_path: Path,
    model_size: str = "1b",
    width: str = "16k",
    l0: str = "medium",
) -> None:
    """Create a report of differential features with Neuronpedia links."""
    lines = []
    lines.append("=" * 70)
    lines.append("SAE FEATURE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Model: gemma-scope-2-{model_size}-pt")
    lines.append(f"Width: {width}, L0: {l0}")
    lines.append("")

    for layer, stats_list in sorted(feature_stats.items()):
        lines.append(f"LAYER {layer}")
        lines.append("-" * 50)

        # Top features by interrupt-complete difference
        lines.append("\nTop features by Interrupt vs Complete difference:")
        sorted_by_ic = sorted(
            stats_list, key=lambda x: abs(x.interrupt_complete_diff), reverse=True
        )

        for i, stat in enumerate(sorted_by_ic[:10], 1):
            url = get_neuronpedia_url(layer, stat.feature_idx, model_size, width, l0)
            lines.append(
                f"  {i}. Feature {stat.feature_idx}: "
                f"diff={stat.interrupt_complete_diff:+.4f} "
                f"(p={stat.interrupt_complete_pvalue:.4f})"
            )
            lines.append(f"     Neuronpedia: {url}")

        # Top features by polite-blunt difference
        lines.append("\nTop features by Polite vs Blunt difference:")
        sorted_by_pb = sorted(
            stats_list, key=lambda x: abs(x.polite_blunt_diff), reverse=True
        )

        for i, stat in enumerate(sorted_by_pb[:10], 1):
            url = get_neuronpedia_url(layer, stat.feature_idx, model_size, width, l0)
            lines.append(
                f"  {i}. Feature {stat.feature_idx}: "
                f"diff={stat.polite_blunt_diff:+.4f} "
                f"(p={stat.polite_blunt_pvalue:.4f})"
            )
            lines.append(f"     Neuronpedia: {url}")

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {output_path}")
