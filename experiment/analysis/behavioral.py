"""Behavioral analysis for conversation friction experiments.

Analyzes accuracy degradation, response patterns, and hedging behavior
across experimental conditions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from experiment.config import (
    Condition,
    COMPLETION_CONDITIONS,
    INTERRUPT_CONDITIONS,
    POLITE_CONDITIONS,
    BLUNT_CONDITIONS,
)


@dataclass
class DegradationAnalysis:
    """Results of degradation slope analysis."""

    condition: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    stderr: float
    num_conversations: int
    num_turns: int


def load_results_to_dataframe(results_path: Path) -> pd.DataFrame:
    """Load conversation results into a pandas DataFrame.

    Args:
        results_path: Path to JSON results file

    Returns:
        DataFrame with one row per turn
    """
    import json

    with open(results_path) as f:
        data = json.load(f)

    rows = []
    for conv in data:
        for turn in conv["turns"]:
            rows.append({
                "conversation_id": conv["conversation_id"],
                "condition": conv["condition"],
                "model_name": conv["model_name"],
                "turn_number": turn["turn_number"],
                "topic": turn["topic"],
                "is_correct": turn["is_correct"],
                "response_length": turn["response_length"],
                "hedging_count": turn["hedging_count"],
                "parsed_answer": turn["parsed_answer"],
            })

    df = pd.DataFrame(rows)

    # Add derived columns
    df["is_interrupt"] = df["condition"].isin(
        [c.value for c in INTERRUPT_CONDITIONS]
    )
    df["is_polite"] = df["condition"].isin(
        [c.value for c in POLITE_CONDITIONS]
    )

    return df


def compute_accuracy_by_turn(
    df: pd.DataFrame,
    group_by: str = "condition",
) -> pd.DataFrame:
    """Compute accuracy at each turn, grouped by condition.

    Args:
        df: DataFrame with turn data
        group_by: Column to group by

    Returns:
        DataFrame with columns [group, turn_number, accuracy, count, std]
    """
    grouped = df.groupby([group_by, "turn_number"])["is_correct"].agg(
        ["mean", "count", "std"]
    ).reset_index()

    grouped.columns = [group_by, "turn_number", "accuracy", "count", "std"]
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])

    return grouped


def fit_degradation_slope(
    df: pd.DataFrame,
    condition: str | None = None,
) -> DegradationAnalysis:
    """Fit linear regression to accuracy over turns.

    Args:
        df: DataFrame with turn data
        condition: Specific condition to analyze (None for all)

    Returns:
        DegradationAnalysis with regression results
    """
    if condition is not None:
        subset = df[df["condition"] == condition]
    else:
        subset = df
        condition = "all"

    # Aggregate accuracy by turn
    by_turn = subset.groupby("turn_number")["is_correct"].mean()

    x = by_turn.index.values
    y = by_turn.values

    # Linear regression
    result = stats.linregress(x, y)

    return DegradationAnalysis(
        condition=condition,
        slope=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue ** 2,
        p_value=result.pvalue,
        stderr=result.stderr,
        num_conversations=subset["conversation_id"].nunique(),
        num_turns=len(x),
    )


def compare_degradation_slopes(
    df: pd.DataFrame,
    group1_conditions: set[Condition],
    group2_conditions: set[Condition],
) -> dict[str, Any]:
    """Compare degradation slopes between two condition groups.

    Args:
        df: DataFrame with turn data
        group1_conditions: First group of conditions
        group2_conditions: Second group of conditions

    Returns:
        Dictionary with comparison statistics
    """
    group1_vals = [c.value for c in group1_conditions]
    group2_vals = [c.value for c in group2_conditions]

    df1 = df[df["condition"].isin(group1_vals)]
    df2 = df[df["condition"].isin(group2_vals)]

    # Get per-conversation slopes
    slopes1 = []
    for conv_id, conv_df in df1.groupby("conversation_id"):
        by_turn = conv_df.groupby("turn_number")["is_correct"].mean()
        if len(by_turn) >= 2:
            result = stats.linregress(by_turn.index, by_turn.values)
            slopes1.append(result.slope)

    slopes2 = []
    for conv_id, conv_df in df2.groupby("conversation_id"):
        by_turn = conv_df.groupby("turn_number")["is_correct"].mean()
        if len(by_turn) >= 2:
            result = stats.linregress(by_turn.index, by_turn.values)
            slopes2.append(result.slope)

    # T-test for difference in slopes
    t_stat, p_value = stats.ttest_ind(slopes1, slopes2)

    return {
        "group1_mean_slope": np.mean(slopes1),
        "group1_std_slope": np.std(slopes1),
        "group2_mean_slope": np.mean(slopes2),
        "group2_std_slope": np.std(slopes2),
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size": (np.mean(slopes1) - np.mean(slopes2)) / np.sqrt(
            (np.var(slopes1) + np.var(slopes2)) / 2
        ),  # Cohen's d
    }


def analyze_hedging_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze hedging behavior across conditions and turns.

    Returns:
        DataFrame with hedging statistics
    """
    # Hedging by condition
    by_condition = df.groupby("condition").agg({
        "hedging_count": ["mean", "std"],
        "response_length": "mean",
    }).reset_index()

    by_condition.columns = [
        "condition",
        "mean_hedging",
        "std_hedging",
        "mean_length",
    ]

    # Normalize hedging by response length
    by_condition["hedging_per_100_chars"] = (
        by_condition["mean_hedging"] / by_condition["mean_length"] * 100
    )

    return by_condition


def run_full_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run complete behavioral analysis.

    Args:
        df: DataFrame with turn data

    Returns:
        Dictionary with all analysis results
    """
    results = {}

    # Accuracy by turn for each condition
    results["accuracy_by_turn"] = compute_accuracy_by_turn(df, "condition")

    # Degradation slopes per condition
    results["degradation_slopes"] = {
        cond.value: fit_degradation_slope(df, cond.value)
        for cond in Condition
    }

    # Compare interrupt vs complete
    results["interrupt_vs_complete"] = compare_degradation_slopes(
        df, INTERRUPT_CONDITIONS, COMPLETION_CONDITIONS
    )

    # Compare polite vs blunt
    results["polite_vs_blunt"] = compare_degradation_slopes(
        df, POLITE_CONDITIONS, BLUNT_CONDITIONS
    )

    # Hedging analysis
    results["hedging"] = analyze_hedging_patterns(df)

    # Overall statistics
    results["summary"] = {
        "total_conversations": df["conversation_id"].nunique(),
        "total_turns": len(df),
        "overall_accuracy": df["is_correct"].mean(),
        "accuracy_by_condition": df.groupby("condition")["is_correct"].mean().to_dict(),
    }

    return results


def plot_accuracy_curves(
    df: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot accuracy degradation curves for each condition.

    Args:
        df: DataFrame with turn data
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: by specific condition
    ax1 = axes[0]
    accuracy_df = compute_accuracy_by_turn(df, "condition")

    for condition in df["condition"].unique():
        cond_data = accuracy_df[accuracy_df["condition"] == condition]
        ax1.errorbar(
            cond_data["turn_number"],
            cond_data["accuracy"],
            yerr=cond_data["se"],
            label=condition,
            marker="o",
            capsize=3,
        )

    ax1.set_xlabel("Turn Number")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Condition")
    ax1.legend(loc="lower left")
    ax1.set_ylim(0, 1)

    # Right plot: interrupt vs complete
    ax2 = axes[1]
    accuracy_by_interrupt = compute_accuracy_by_turn(df, "is_interrupt")

    for is_interrupt in [True, False]:
        label = "Interrupt" if is_interrupt else "Complete"
        data = accuracy_by_interrupt[accuracy_by_interrupt["is_interrupt"] == is_interrupt]
        ax2.errorbar(
            data["turn_number"],
            data["accuracy"],
            yerr=data["se"],
            label=label,
            marker="o",
            capsize=3,
            linewidth=2,
        )

    ax2.set_xlabel("Turn Number")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy: Interrupt vs Complete")
    ax2.legend(loc="lower left")
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_degradation_comparison(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot degradation slope comparison between groups.

    Args:
        df: DataFrame with turn data
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect per-conversation slopes
    def get_slopes(subset_df: pd.DataFrame) -> list[float]:
        slopes = []
        for conv_id, conv_df in subset_df.groupby("conversation_id"):
            by_turn = conv_df.groupby("turn_number")["is_correct"].mean()
            if len(by_turn) >= 2:
                result = stats.linregress(by_turn.index, by_turn.values)
                slopes.append(result.slope)
        return slopes

    # Interrupt vs Complete
    ax1 = axes[0]
    interrupt_slopes = get_slopes(df[df["is_interrupt"] == True])
    complete_slopes = get_slopes(df[df["is_interrupt"] == False])

    data_to_plot = pd.DataFrame({
        "Slope": interrupt_slopes + complete_slopes,
        "Type": ["Interrupt"] * len(interrupt_slopes) + ["Complete"] * len(complete_slopes),
    })

    sns.boxplot(data=data_to_plot, x="Type", y="Slope", ax=ax1)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Degradation Slope: Interrupt vs Complete")
    ax1.set_ylabel("Slope (accuracy/turn)")

    # Polite vs Blunt
    ax2 = axes[1]
    polite_slopes = get_slopes(df[df["is_polite"] == True])
    blunt_slopes = get_slopes(df[df["is_polite"] == False])

    data_to_plot = pd.DataFrame({
        "Slope": polite_slopes + blunt_slopes,
        "Type": ["Polite"] * len(polite_slopes) + ["Blunt"] * len(blunt_slopes),
    })

    sns.boxplot(data=data_to_plot, x="Type", y="Slope", ax=ax2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Degradation Slope: Polite vs Blunt")
    ax2.set_ylabel("Slope (accuracy/turn)")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def generate_report(
    analysis_results: dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a text report of behavioral analysis.

    Args:
        analysis_results: Results from run_full_analysis
        output_path: Path to save report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BEHAVIORAL ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    summary = analysis_results["summary"]
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total conversations: {summary['total_conversations']}")
    lines.append(f"Total turns: {summary['total_turns']}")
    lines.append(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
    lines.append("")

    lines.append("Accuracy by condition:")
    for cond, acc in sorted(summary["accuracy_by_condition"].items()):
        lines.append(f"  {cond}: {acc:.3f}")
    lines.append("")

    # Degradation slopes
    lines.append("DEGRADATION SLOPES")
    lines.append("-" * 40)
    for cond, deg in analysis_results["degradation_slopes"].items():
        lines.append(
            f"{cond}: slope={deg.slope:.4f}, "
            f"R²={deg.r_squared:.3f}, p={deg.p_value:.4f}"
        )
    lines.append("")

    # Comparisons
    lines.append("GROUP COMPARISONS")
    lines.append("-" * 40)

    ivsc = analysis_results["interrupt_vs_complete"]
    lines.append("Interrupt vs Complete:")
    lines.append(f"  Interrupt mean slope: {ivsc['group1_mean_slope']:.4f} (±{ivsc['group1_std_slope']:.4f})")
    lines.append(f"  Complete mean slope: {ivsc['group2_mean_slope']:.4f} (±{ivsc['group2_std_slope']:.4f})")
    lines.append(f"  t-statistic: {ivsc['t_statistic']:.3f}")
    lines.append(f"  p-value: {ivsc['p_value']:.4f}")
    lines.append(f"  Effect size (Cohen's d): {ivsc['effect_size']:.3f}")
    lines.append("")

    pvsb = analysis_results["polite_vs_blunt"]
    lines.append("Polite vs Blunt:")
    lines.append(f"  Polite mean slope: {pvsb['group1_mean_slope']:.4f} (±{pvsb['group1_std_slope']:.4f})")
    lines.append(f"  Blunt mean slope: {pvsb['group2_mean_slope']:.4f} (±{pvsb['group2_std_slope']:.4f})")
    lines.append(f"  t-statistic: {pvsb['t_statistic']:.3f}")
    lines.append(f"  p-value: {pvsb['p_value']:.4f}")
    lines.append(f"  Effect size (Cohen's d): {pvsb['effect_size']:.3f}")
    lines.append("")

    # Hedging
    lines.append("HEDGING ANALYSIS")
    lines.append("-" * 40)
    hedging = analysis_results["hedging"]
    for _, row in hedging.iterrows():
        lines.append(
            f"{row['condition']}: {row['mean_hedging']:.2f} markers/response "
            f"({row['hedging_per_100_chars']:.2f} per 100 chars)"
        )
    lines.append("")

    lines.append("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {output_path}")
