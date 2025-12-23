"""MMLU-Pro dataset loading and question management."""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from datasets import load_dataset, Dataset


@dataclass
class Question:
    """A single MMLU-Pro question."""

    question_id: str
    category: str
    question: str
    options: list[str]  # 10 options (A-J)
    answer: str  # letter (A-J)
    answer_index: int  # 0-9

    def format_for_prompt(self) -> str:
        """Format question for inclusion in a prompt."""
        options_str = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(self.options)
        )
        return f"{self.question}\n\n{options_str}"

    def format_with_instruction(self) -> str:
        """Format question with answering instruction."""
        return (
            f"{self.format_for_prompt()}\n\n"
            "Please provide your answer as a single letter (A-J)."
        )


class MMLUProDataset:
    """Manager for MMLU-Pro dataset organized by category."""

    def __init__(self, split: str = "test", seed: int = 42):
        """Load MMLU-Pro dataset.

        Args:
            split: Dataset split to use ("test" or "validation")
            seed: Random seed for shuffling
        """
        self.split = split
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Load dataset
        self._dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)

        # Organize by category
        self._by_category: dict[str, list[Question]] = {}
        self._build_category_index()

        # Track which questions have been used (per category)
        self._used_indices: dict[str, set[int]] = {
            cat: set() for cat in self._by_category
        }

    def _build_category_index(self) -> None:
        """Build index of questions by category."""
        for idx, item in enumerate(self._dataset):
            category = item["category"].lower()

            if category not in self._by_category:
                self._by_category[category] = []

            question = Question(
                question_id=f"{category}_{idx}",
                category=category,
                question=item["question"],
                options=item["options"],
                answer=item["answer"],
                answer_index=item["answer_index"],
            )
            self._by_category[category].append(question)

        # Shuffle questions within each category
        for category in self._by_category:
            self.rng.shuffle(self._by_category[category])

    @property
    def categories(self) -> list[str]:
        """Get list of available categories."""
        return sorted(self._by_category.keys())

    def num_questions(self, category: str) -> int:
        """Get number of questions in a category."""
        category = category.lower()
        return len(self._by_category.get(category, []))

    def get_question(self, category: str, avoid_repeat: bool = True) -> Question | None:
        """Get a random question from a category.

        Args:
            category: Category name
            avoid_repeat: If True, avoid returning previously used questions

        Returns:
            Question object, or None if no questions available
        """
        category = category.lower()
        if category not in self._by_category:
            raise ValueError(f"Unknown category: {category}")

        questions = self._by_category[category]
        used = self._used_indices[category]

        if avoid_repeat:
            available = [i for i in range(len(questions)) if i not in used]
            if not available:
                # Reset if we've used all questions
                self._used_indices[category] = set()
                available = list(range(len(questions)))

            idx = self.rng.choice(available)
            self._used_indices[category].add(idx)
            return questions[idx]
        else:
            return self.rng.choice(questions)

    def get_questions_for_conversation(
        self,
        topics: list[str],
        num_per_topic: int = 1,
    ) -> list[Question]:
        """Get questions for a multi-turn conversation.

        Args:
            topics: List of topic/category names (one per turn)
            num_per_topic: Number of questions to get per topic (usually 1)

        Returns:
            List of questions in order
        """
        questions = []
        for topic in topics:
            for _ in range(num_per_topic):
                q = self.get_question(topic)
                if q is not None:
                    questions.append(q)
        return questions

    def reset_usage(self, category: str | None = None) -> None:
        """Reset usage tracking.

        Args:
            category: Specific category to reset, or None for all
        """
        if category is None:
            self._used_indices = {cat: set() for cat in self._by_category}
        else:
            self._used_indices[category.lower()] = set()

    def get_category_stats(self) -> dict[str, int]:
        """Get number of questions per category."""
        return {cat: len(qs) for cat, qs in sorted(self._by_category.items())}


def parse_model_answer(response: str) -> str | None:
    """Extract the answer letter from a model response.

    Handles various response formats:
    - "A"
    - "The answer is A"
    - "A. option text"
    - etc.

    Returns:
        Single letter A-J, or None if couldn't parse
    """
    response = response.strip().upper()

    # Direct single letter
    if len(response) == 1 and response in "ABCDEFGHIJ":
        return response

    # Look for patterns like "answer is X" or "answer: X"
    import re

    patterns = [
        r"(?:the\s+)?answer\s+is\s+([A-J])",
        r"(?:the\s+)?answer:\s*([A-J])",
        r"^([A-J])\.",  # starts with "A."
        r"^([A-J])\s",  # starts with "A "
        r"\b([A-J])\b",  # any standalone letter
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def check_answer(question: Question, response: str) -> tuple[bool, str | None]:
    """Check if model response is correct.

    Returns:
        Tuple of (is_correct, parsed_answer)
    """
    parsed = parse_model_answer(response)
    if parsed is None:
        return False, None
    return parsed == question.answer, parsed


if __name__ == "__main__":
    # Demo: load dataset and show statistics
    print("Loading MMLU-Pro dataset...")
    dataset = MMLUProDataset(split="test", seed=42)

    print("\nCategory statistics:")
    stats = dataset.get_category_stats()
    for cat, count in stats.items():
        print(f"  {cat}: {count} questions")

    print(f"\nTotal questions: {sum(stats.values())}")

    # Show sample question
    print("\nSample question from 'physics':")
    q = dataset.get_question("physics")
    if q:
        print(q.format_with_instruction())
        print(f"\nCorrect answer: {q.answer}")
