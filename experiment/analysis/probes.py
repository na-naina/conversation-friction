"""Linear probe training for condition classification.

Trains classifiers to predict experimental condition from activations,
testing whether condition information is linearly readable.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from experiment.activation_hooks import TurnActivations
from experiment.config import (
    INTERRUPT_CONDITIONS,
    COMPLETION_CONDITIONS,
)


@dataclass
class ProbeResults:
    """Results from training a linear probe."""

    task: str  # e.g., "interrupt_vs_complete"
    layer: int
    accuracy: float
    accuracy_std: float  # from cross-validation
    balanced_accuracy: float
    coefficients: np.ndarray | None  # probe weights
    classes: list[str]
    confusion_matrix: np.ndarray
    classification_report: str


def load_activations_for_probing(
    activations_dir: Path,
    layer: int,
) -> tuple[Tensor, list[str], list[str]]:
    """Load activations and labels for probe training.

    Args:
        activations_dir: Directory with saved TurnActivations
        layer: Which layer to load

    Returns:
        Tuple of (activations, conditions, conversation_ids)
    """
    activations = []
    conditions = []
    conv_ids = []

    for conv_dir in activations_dir.iterdir():
        if not conv_dir.is_dir():
            continue

        for turn_file in sorted(conv_dir.glob("turn_*.pt")):
            turn_acts = TurnActivations.load(turn_file)

            if layer not in turn_acts.layer_activations:
                continue

            # Average over tokens
            raw_acts = turn_acts.layer_activations[layer]
            mean_act = raw_acts.mean(dim=0)

            activations.append(mean_act)
            conditions.append(turn_acts.condition)
            conv_ids.append(turn_acts.conversation_id)

    return torch.stack(activations), conditions, conv_ids


def create_binary_labels(
    conditions: list[str],
    task: Literal["interrupt_vs_complete", "polite_vs_blunt"],
) -> tuple[np.ndarray, list[str]]:
    """Create binary labels for a classification task.

    Args:
        conditions: List of condition strings
        task: Which binary task to create labels for

    Returns:
        Tuple of (labels as 0/1, class names)
    """
    if task == "interrupt_vs_complete":
        interrupt_vals = {c.value for c in INTERRUPT_CONDITIONS}
        labels = np.array([1 if c in interrupt_vals else 0 for c in conditions])
        classes = ["complete", "interrupt"]
    elif task == "polite_vs_blunt":
        polite_vals = {"complete_grateful", "interrupt_polite"}
        labels = np.array([1 if c in polite_vals else 0 for c in conditions])
        classes = ["blunt", "polite"]
    else:
        raise ValueError(f"Unknown task: {task}")

    return labels, classes


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    task: str,
    layer: int,
    n_cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ProbeResults:
    """Train a linear probe with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        classes: Class names
        task: Task name for logging
        layer: Layer index
        n_cv_folds: Number of cross-validation folds
        test_size: Fraction of data for test set
        random_state: Random seed

    Returns:
        ProbeResults with accuracy metrics
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation for accuracy estimate
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )

    cv_scores = cross_val_score(clf, X_scaled, y, cv=n_cv_folds, scoring="accuracy")

    # Train final model for coefficients and detailed metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute metrics
    from sklearn.metrics import balanced_accuracy_score

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes)

    return ProbeResults(
        task=task,
        layer=layer,
        accuracy=float(cv_scores.mean()),
        accuracy_std=float(cv_scores.std()),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        coefficients=clf.coef_,
        classes=classes,
        confusion_matrix=cm,
        classification_report=report,
    )


def train_probes_all_layers(
    activations_dir: Path,
    layers: list[int],
    tasks: list[Literal["interrupt_vs_complete", "polite_vs_blunt"]] | None = None,
) -> dict[str, dict[int, ProbeResults]]:
    """Train probes for all layers and tasks.

    Args:
        activations_dir: Directory with saved activations
        layers: Which layers to probe
        tasks: Which tasks to train (defaults to both)

    Returns:
        Dict mapping task -> layer -> ProbeResults
    """
    if tasks is None:
        tasks = ["interrupt_vs_complete", "polite_vs_blunt"]

    results: dict[str, dict[int, ProbeResults]] = {task: {} for task in tasks}

    for layer in layers:
        print(f"Loading activations for layer {layer}...")
        X_tensor, conditions, _ = load_activations_for_probing(activations_dir, layer)
        X = X_tensor.numpy()

        for task in tasks:
            print(f"  Training {task} probe...")
            y, classes = create_binary_labels(conditions, task)

            probe_result = train_probe(X, y, classes, task, layer)
            results[task][layer] = probe_result

            print(
                f"    Accuracy: {probe_result.accuracy:.3f} "
                f"(±{probe_result.accuracy_std:.3f})"
            )

    return results


def extract_probe_direction(
    probe_result: ProbeResults,
) -> np.ndarray:
    """Extract the classification direction from a trained probe.

    This direction can be used for activation steering experiments.

    Args:
        probe_result: Trained probe results

    Returns:
        Unit vector in the classification direction
    """
    if probe_result.coefficients is None:
        raise ValueError("Probe has no coefficients (not trained?)")

    # For binary classification, coefficients shape is (1, n_features)
    direction = probe_result.coefficients[0]

    # Normalize to unit vector
    direction = direction / np.linalg.norm(direction)

    return direction


def analyze_probe_accuracy_vs_turn(
    activations_dir: Path,
    layer: int,
    task: Literal["interrupt_vs_complete", "polite_vs_blunt"],
) -> dict[int, float]:
    """Analyze how probe accuracy varies by turn number.

    Useful for understanding when condition effects emerge.

    Args:
        activations_dir: Directory with activations
        layer: Layer to analyze
        task: Classification task

    Returns:
        Dict mapping turn_number -> accuracy
    """
    # Load activations grouped by turn
    activations_by_turn: dict[int, list[Tensor]] = {}
    conditions_by_turn: dict[int, list[str]] = {}

    for conv_dir in activations_dir.iterdir():
        if not conv_dir.is_dir():
            continue

        for turn_file in sorted(conv_dir.glob("turn_*.pt")):
            turn_acts = TurnActivations.load(turn_file)

            if layer not in turn_acts.layer_activations:
                continue

            turn = turn_acts.turn_number
            if turn not in activations_by_turn:
                activations_by_turn[turn] = []
                conditions_by_turn[turn] = []

            raw_acts = turn_acts.layer_activations[layer]
            mean_act = raw_acts.mean(dim=0)

            activations_by_turn[turn].append(mean_act)
            conditions_by_turn[turn].append(turn_acts.condition)

    # Train probe for each turn
    results = {}

    for turn in sorted(activations_by_turn.keys()):
        X = torch.stack(activations_by_turn[turn]).numpy()
        y, classes = create_binary_labels(conditions_by_turn[turn], task)

        # Simple accuracy (no cross-val due to smaller sample)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

        # Use cross-val if enough samples, otherwise simple split
        if len(X) >= 20:
            scores = cross_val_score(clf, X_scaled, y, cv=5)
            results[turn] = float(scores.mean())
        else:
            clf.fit(X_scaled, y)
            results[turn] = float(clf.score(X_scaled, y))

    return results


def plot_probe_results(
    results: dict[str, dict[int, ProbeResults]],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot probe accuracy across layers.

    Args:
        results: Results from train_probes_all_layers
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for task, layer_results in results.items():
        layers = sorted(layer_results.keys())
        accuracies = [layer_results[l].accuracy for l in layers]
        stds = [layer_results[l].accuracy_std for l in layers]

        ax.errorbar(
            layers, accuracies, yerr=stds,
            label=task.replace("_", " ").title(),
            marker="o", capsize=3,
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Linear Probe Accuracy by Layer")
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_accuracy_by_turn(
    accuracy_by_turn: dict[int, float],
    task: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot probe accuracy by turn number.

    Args:
        accuracy_by_turn: Results from analyze_probe_accuracy_vs_turn
        task: Task name for title
        output_path: Optional save path

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    turns = sorted(accuracy_by_turn.keys())
    accuracies = [accuracy_by_turn[t] for t in turns]

    ax.plot(turns, accuracies, marker="o", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title(f"Probe Accuracy by Turn: {task.replace('_', ' ').title()}")
    ax.set_ylim(0.4, 1.0)
    ax.legend()

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def generate_probe_report(
    results: dict[str, dict[int, ProbeResults]],
    output_path: Path,
) -> None:
    """Generate text report of probe results.

    Args:
        results: Results from train_probes_all_layers
        output_path: Path to save report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("LINEAR PROBE ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    for task, layer_results in results.items():
        lines.append(f"TASK: {task.upper().replace('_', ' ')}")
        lines.append("-" * 40)

        # Find best layer
        best_layer = max(layer_results.keys(), key=lambda l: layer_results[l].accuracy)
        best_acc = layer_results[best_layer].accuracy

        lines.append(f"Best layer: {best_layer} (accuracy: {best_acc:.3f})")
        lines.append("")

        lines.append("Accuracy by layer:")
        for layer in sorted(layer_results.keys()):
            res = layer_results[layer]
            lines.append(
                f"  Layer {layer}: {res.accuracy:.3f} (±{res.accuracy_std:.3f}), "
                f"balanced: {res.balanced_accuracy:.3f}"
            )

        lines.append("")

        # Best layer details
        best_res = layer_results[best_layer]
        lines.append(f"Classification report for layer {best_layer}:")
        lines.append(best_res.classification_report)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {output_path}")
