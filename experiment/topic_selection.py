"""Programmatic topic selection using embedding distance.

Selects MMLU-Pro categories that are maximally dissimilar to ensure
topic independence across conversation turns.
"""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from experiment.config import MMLU_PRO_CATEGORIES


def compute_category_embeddings(
    categories: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> NDArray[np.float32]:
    """Compute embeddings for category names.

    Args:
        categories: List of category names to embed
        model_name: Sentence transformer model to use

    Returns:
        Array of shape (num_categories, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(categories, convert_to_numpy=True)
    return embeddings  # type: ignore[return-value]


def compute_pairwise_distances(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute pairwise cosine distances between embeddings.

    Args:
        embeddings: Array of shape (n, d)

    Returns:
        Distance matrix of shape (n, n) where distance = 1 - cosine_similarity
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Cosine similarity
    similarity = normalized @ normalized.T

    # Convert to distance
    distance = 1 - similarity
    return distance.astype(np.float32)


def select_diverse_topics(
    num_topics: int,
    min_distance: float = 0.3,
    categories: list[str] | None = None,
    seed: int = 42,
) -> list[str]:
    """Select maximally diverse topics using greedy farthest-point sampling.

    Algorithm:
    1. Start with a random topic
    2. Iteratively add the topic that maximizes minimum distance to selected set
    3. Stop when we have enough topics or can't find topics above min_distance

    Args:
        num_topics: Number of topics to select
        min_distance: Minimum cosine distance between any pair of selected topics
        categories: List of categories to choose from (defaults to MMLU_PRO_CATEGORIES)
        seed: Random seed for reproducibility

    Returns:
        List of selected category names
    """
    if categories is None:
        categories = MMLU_PRO_CATEGORIES

    if num_topics > len(categories):
        raise ValueError(
            f"Requested {num_topics} topics but only {len(categories)} categories available"
        )

    rng = np.random.default_rng(seed)

    # Compute embeddings and distances
    embeddings = compute_category_embeddings(categories)
    distances = compute_pairwise_distances(embeddings)

    # Greedy farthest-point sampling
    selected_indices: list[int] = []
    available_indices = set(range(len(categories)))

    # Start with random topic
    first_idx = rng.choice(list(available_indices))
    selected_indices.append(first_idx)
    available_indices.remove(first_idx)

    while len(selected_indices) < num_topics and available_indices:
        # Find the available topic with maximum minimum distance to selected set
        best_idx = -1
        best_min_dist = -1.0

        for idx in available_indices:
            # Minimum distance to any selected topic
            min_dist = min(distances[idx, sel_idx] for sel_idx in selected_indices)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx

        # Check if best topic meets minimum distance requirement
        if best_min_dist < min_distance:
            print(
                f"Warning: Could only select {len(selected_indices)} topics "
                f"with min_distance >= {min_distance}. "
                f"Best available distance: {best_min_dist:.3f}"
            )
            break

        selected_indices.append(best_idx)
        available_indices.remove(best_idx)

    selected_topics = [categories[i] for i in selected_indices]
    return selected_topics


def create_topic_sequence(
    topics: list[str],
    num_turns: int,
    seed: int = 42,
) -> list[str]:
    """Create a sequence of topics for a conversation.

    If num_turns > len(topics), topics are repeated in a shuffled order.
    This ensures no two consecutive turns have the same topic.

    Args:
        topics: List of available topics
        num_turns: Number of turns in the conversation
        seed: Random seed

    Returns:
        List of topic names for each turn
    """
    rng = np.random.default_rng(seed)

    sequence = []
    remaining = list(topics)
    rng.shuffle(remaining)

    for _ in range(num_turns):
        if not remaining:
            # Reshuffle, but avoid repeating the last topic
            remaining = list(topics)
            if sequence and remaining[0] == sequence[-1]:
                # Swap first element with a random other element
                swap_idx = rng.integers(1, len(remaining))
                remaining[0], remaining[swap_idx] = remaining[swap_idx], remaining[0]
            rng.shuffle(remaining)

        sequence.append(remaining.pop())

    return sequence


def get_topic_statistics(topics: list[str]) -> dict[str, float]:
    """Compute statistics about topic selection.

    Returns minimum, maximum, and mean pairwise distances.
    """
    embeddings = compute_category_embeddings(topics)
    distances = compute_pairwise_distances(embeddings)

    # Get upper triangle (excluding diagonal)
    n = len(topics)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_distances = distances[upper_tri_indices]

    return {
        "min_distance": float(pairwise_distances.min()),
        "max_distance": float(pairwise_distances.max()),
        "mean_distance": float(pairwise_distances.mean()),
        "std_distance": float(pairwise_distances.std()),
    }


if __name__ == "__main__":
    # Demo: select diverse topics
    print("MMLU-Pro Categories:")
    for cat in MMLU_PRO_CATEGORIES:
        print(f"  - {cat}")

    print("\nSelecting 12 maximally diverse topics...")
    selected = select_diverse_topics(num_topics=12, min_distance=0.2, seed=42)

    print("\nSelected topics:")
    for i, topic in enumerate(selected, 1):
        print(f"  {i}. {topic}")

    stats = get_topic_statistics(selected)
    print(f"\nTopic diversity statistics:")
    print(f"  Min pairwise distance: {stats['min_distance']:.3f}")
    print(f"  Max pairwise distance: {stats['max_distance']:.3f}")
    print(f"  Mean pairwise distance: {stats['mean_distance']:.3f}")
