"""Main experiment runner.

Entry point for running the conversation friction experiment.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from experiment.config import ExperimentConfig, Condition
from experiment.topic_selection import select_diverse_topics, get_topic_statistics
from experiment.dataset import MMLUProDataset
from experiment.conversation import (
    run_experiment_batch,
    save_results,
    load_results,
)
from experiment.analysis.behavioral import (
    load_results_to_dataframe,
    run_full_analysis,
    plot_accuracy_curves,
    plot_degradation_comparison,
    generate_report,
)


def run_behavioral_experiment(
    config: ExperimentConfig,
    output_dir: Path | None = None,
) -> Path:
    """Run the full behavioral experiment.

    Args:
        config: Experiment configuration
        output_dir: Override output directory

    Returns:
        Path to results file
    """
    print("=" * 60)
    print("CONVERSATION FRICTION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {config.model_config.name}")
    print(f"Turns per conversation: {config.num_turns}")
    print(f"Conversations per condition: {config.num_conversations_per_condition}")
    print()

    # Ensure directories exist
    config.ensure_dirs()
    results_dir = output_dir or config.results_dir

    # Step 1: Select diverse topics
    print("Step 1: Selecting diverse topics...")
    topics = select_diverse_topics(
        num_topics=config.num_topics,
        min_distance=config.min_topic_distance,
        seed=config.seed,
    )
    print(f"Selected {len(topics)} topics: {', '.join(topics)}")

    stats = get_topic_statistics(topics)
    print(f"Topic diversity - min distance: {stats['min_distance']:.3f}")
    print()

    # Step 2: Load dataset
    print("Step 2: Loading MMLU-Pro dataset...")
    dataset = MMLUProDataset(split="test", seed=config.seed)
    print(f"Loaded {sum(dataset.get_category_stats().values())} questions")
    print()

    # Step 3: Run conversations
    print("Step 3: Running conversations...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"results_{timestamp}.json"
    checkpoint_path = results_dir / f"checkpoint_{timestamp}.json"

    results = run_experiment_batch(
        config=config,
        dataset=dataset,
        topics=topics,
        checkpoint_path=checkpoint_path,
        checkpoint_every=10,  # Save every 10 conversations
    )

    # Step 4: Save final results
    save_results(results, results_path)
    print()

    # Step 5: Quick summary
    print("Quick Summary:")
    for condition in Condition:
        cond_results = [r for r in results if r.condition == condition.value]
        if cond_results:
            avg_acc = sum(r.overall_accuracy for r in cond_results) / len(cond_results)
            print(f"  {condition.value}: {avg_acc:.3f} accuracy")

    return results_path


def run_analysis(results_path: Path, output_dir: Path | None = None) -> None:
    """Run analysis on experiment results.

    Args:
        results_path: Path to results JSON file
        output_dir: Output directory for figures and reports
    """
    print("=" * 60)
    print("BEHAVIORAL ANALYSIS")
    print("=" * 60)

    if output_dir is None:
        output_dir = results_path.parent

    # Load data
    print("Loading results...")
    df = load_results_to_dataframe(results_path)
    print(f"Loaded {len(df)} turns from {df['conversation_id'].nunique()} conversations")
    print()

    # Run analysis
    print("Running analysis...")
    analysis = run_full_analysis(df)

    # Generate report
    report_path = output_dir / "behavioral_report.txt"
    generate_report(analysis, report_path)

    # Generate figures
    print("Generating figures...")
    accuracy_fig = plot_accuracy_curves(df, output_dir / "accuracy_curves.png")
    degradation_fig = plot_degradation_comparison(df, output_dir / "degradation_comparison.png")

    # Print key findings
    print()
    print("KEY FINDINGS:")
    print("-" * 40)

    ivsc = analysis["interrupt_vs_complete"]
    print(f"Interrupt vs Complete effect:")
    print(f"  Slope difference: {ivsc['group1_mean_slope'] - ivsc['group2_mean_slope']:.4f}")
    print(f"  p-value: {ivsc['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {ivsc['effect_size']:.3f}")

    print()
    pvsb = analysis["polite_vs_blunt"]
    print(f"Polite vs Blunt effect:")
    print(f"  Slope difference: {pvsb['group1_mean_slope'] - pvsb['group2_mean_slope']:.4f}")
    print(f"  p-value: {pvsb['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {pvsb['effect_size']:.3f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run conversation friction experiment"
    )
    parser.add_argument(
        "--mode",
        choices=["run", "analyze", "both"],
        default="both",
        help="Run experiment, analyze results, or both",
    )
    parser.add_argument(
        "--model-size",
        choices=["1b", "4b", "12b", "27b"],
        default="4b",
        help="Model size to use (1b fits ~6GB VRAM, 4b needs ~10GB)",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=14,
        help="Number of turns per conversation (max 14 due to MMLU-Pro categories)",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=50,
        help="Number of conversations per condition",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help="Path to existing results (for analyze mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Create config
    # Cap num_turns at 14 (number of MMLU-Pro categories)
    num_turns = min(args.num_turns, 14)
    config = ExperimentConfig(
        model_size=args.model_size,
        num_turns=num_turns,
        num_topics=num_turns,  # One topic per turn
        num_conversations_per_condition=args.num_conversations,
        seed=args.seed,
    )

    if args.mode in ["run", "both"]:
        results_path = run_behavioral_experiment(config, args.output_dir)
    else:
        results_path = args.results_path

    if args.mode in ["analyze", "both"]:
        if results_path is None:
            parser.error("--results-path required for analyze mode")
        run_analysis(results_path, args.output_dir)


if __name__ == "__main__":
    main()
