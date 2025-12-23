"""SAE feature analysis for conversation friction experiments.

Loads Gemma Scope 2 SAEs and analyzes which features activate differentially
across experimental conditions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import pandas as pd
from tqdm import tqdm
from scipy import stats


@dataclass
class SAEConfig:
    """Configuration for loading Gemma Scope SAEs."""

    repo_id: str  # HuggingFace repo
    layer: int
    width: str = "16k"  # SAE width (16k or 131k typically)


class GemmaScopeSAE:
    """Wrapper for Gemma Scope SAE loading and encoding."""

    def __init__(
        self,
        repo_id: str,
        layer: int,
        width: str = "16k",
        device: str = "cuda",
    ):
        """Load SAE from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            layer: Which layer's SAE to load
            width: SAE width (16k or 131k)
            device: Device to load to
        """
        self.repo_id = repo_id
        self.layer = layer
        self.width = width
        self.device = device

        # Load SAE parameters
        self._load_sae()

    def _load_sae(self) -> None:
        """Load SAE weights from HuggingFace."""
        from huggingface_hub import hf_hub_download

        # Download SAE weights
        # Gemma Scope 2 structure: layer_X/width_Yk/...
        subfolder = f"layer_{self.layer}/width_{self.width}"

        try:
            # Try to load params.npz (common format)
            params_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="params.npz",
                subfolder=subfolder,
            )
            params = np.load(params_path)

            self.W_enc = torch.from_numpy(params["W_enc"]).to(self.device)
            self.W_dec = torch.from_numpy(params["W_dec"]).to(self.device)
            self.b_enc = torch.from_numpy(params["b_enc"]).to(self.device)
            self.b_dec = torch.from_numpy(params["b_dec"]).to(self.device)

        except Exception as e:
            # Try alternative format (safetensors or pt)
            try:
                from safetensors.torch import load_file

                sae_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename="sae_weights.safetensors",
                    subfolder=subfolder,
                )
                state_dict = load_file(sae_path)

                self.W_enc = state_dict["W_enc"].to(self.device)
                self.W_dec = state_dict["W_dec"].to(self.device)
                self.b_enc = state_dict.get("b_enc", torch.zeros(self.W_enc.shape[1])).to(self.device)
                self.b_dec = state_dict.get("b_dec", torch.zeros(self.W_dec.shape[1])).to(self.device)

            except Exception as e2:
                raise ValueError(
                    f"Could not load SAE from {self.repo_id}/{subfolder}. "
                    f"Errors: {e}, {e2}"
                )

        self.d_model = self.W_enc.shape[0]
        self.n_features = self.W_enc.shape[1]

    def encode(self, activations: Tensor) -> Tensor:
        """Encode activations to SAE feature space.

        Args:
            activations: Tensor of shape (..., d_model)

        Returns:
            Feature activations of shape (..., n_features)
        """
        # Pre-encoder bias (center data)
        centered = activations - self.b_dec

        # Encode
        pre_acts = centered @ self.W_enc + self.b_enc

        # ReLU activation
        feature_acts = torch.relu(pre_acts)

        return feature_acts

    def decode(self, feature_acts: Tensor) -> Tensor:
        """Decode from feature space back to activation space.

        Args:
            feature_acts: Tensor of shape (..., n_features)

        Returns:
            Reconstructed activations of shape (..., d_model)
        """
        return feature_acts @ self.W_dec + self.b_dec


def load_sae_for_layer(
    sae_repo: str,
    layer: int,
    width: str = "16k",
    device: str = "cuda",
) -> GemmaScopeSAE:
    """Convenience function to load SAE for a specific layer.

    Args:
        sae_repo: HuggingFace repo ID
        layer: Layer index
        width: SAE width
        device: Device

    Returns:
        Loaded SAE
    """
    return GemmaScopeSAE(
        repo_id=sae_repo,
        layer=layer,
        width=width,
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
    sae_repo: str,
    layers: list[int],
    top_k: int = 50,
    device: str = "cuda",
) -> dict[int, list[FeatureActivationStats]]:
    """Find features that activate differentially across conditions.

    Args:
        activations_dir: Directory with saved activations
        sae_repo: HuggingFace repo for SAEs
        layers: Which layers to analyze
        top_k: Number of top features per layer
        device: Device for SAE

    Returns:
        Dict mapping layer -> list of top differential features
    """
    results = {}

    for layer in tqdm(layers, desc="Analyzing layers"):
        # Load SAE
        sae = load_sae_for_layer(sae_repo, layer, device=device)

        # Encode activations
        feature_acts = encode_activations_with_sae(activations_dir, sae, layer)

        # Compute statistics
        stats = compute_feature_statistics(feature_acts, layer, top_k)
        results[layer] = stats

    return results


def features_to_dataframe(
    feature_stats: dict[int, list[FeatureActivationStats]],
) -> pd.DataFrame:
    """Convert feature statistics to DataFrame.

    Args:
        feature_stats: Dict from find_differential_features

    Returns:
        DataFrame with feature statistics
    """
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
            # Add per-condition means
            for cond, mean in stat.mean_by_condition.items():
                row[f"mean_{cond}"] = mean
            rows.append(row)

    return pd.DataFrame(rows)


def get_neuronpedia_url(
    layer: int,
    feature_idx: int,
    model_name: str = "gemma-3-4b",
    sae_id: str = "res-16k",
) -> str:
    """Generate Neuronpedia URL for a feature.

    Args:
        layer: Layer index
        feature_idx: Feature index
        model_name: Model name on Neuronpedia
        sae_id: SAE identifier

    Returns:
        URL string
    """
    return (
        f"https://neuronpedia.org/{model_name}/"
        f"{layer}-{sae_id}/{feature_idx}"
    )


def create_feature_report(
    feature_stats: dict[int, list[FeatureActivationStats]],
    output_path: Path,
    model_name: str = "gemma-3-4b",
) -> None:
    """Create a report of differential features with Neuronpedia links.

    Args:
        feature_stats: Results from find_differential_features
        output_path: Path to save report
        model_name: Model name for Neuronpedia URLs
    """
    lines = []
    lines.append("=" * 70)
    lines.append("SAE FEATURE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    for layer, stats_list in sorted(feature_stats.items()):
        lines.append(f"LAYER {layer}")
        lines.append("-" * 50)

        # Top features by interrupt-complete difference
        lines.append("\nTop features by Interrupt vs Complete difference:")
        sorted_by_ic = sorted(stats_list, key=lambda x: abs(x.interrupt_complete_diff), reverse=True)

        for i, stat in enumerate(sorted_by_ic[:10], 1):
            url = get_neuronpedia_url(layer, stat.feature_idx, model_name)
            lines.append(
                f"  {i}. Feature {stat.feature_idx}: "
                f"diff={stat.interrupt_complete_diff:+.4f} "
                f"(p={stat.interrupt_complete_pvalue:.4f})"
            )
            lines.append(f"     Neuronpedia: {url}")

        # Top features by polite-blunt difference
        lines.append("\nTop features by Polite vs Blunt difference:")
        sorted_by_pb = sorted(stats_list, key=lambda x: abs(x.polite_blunt_diff), reverse=True)

        for i, stat in enumerate(sorted_by_pb[:10], 1):
            url = get_neuronpedia_url(layer, stat.feature_idx, model_name)
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
