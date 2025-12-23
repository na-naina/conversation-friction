"""Conversation runner for multi-turn experiments.

Manages the conversation flow, applies condition templates, and collects
responses from the model.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from experiment.config import (
    Condition,
    CONDITION_TEMPLATES,
    ExperimentConfig,
    INTERRUPT_CONDITIONS,
)
from experiment.dataset import Question, MMLUProDataset, check_answer


@dataclass
class TurnResult:
    """Result of a single conversation turn."""

    turn_number: int
    topic: str
    question_id: str
    question_text: str
    correct_answer: str
    model_response: str
    parsed_answer: str | None
    is_correct: bool
    response_length: int
    hedging_count: int
    # Whether this response was truncated in conversation history
    was_truncated: bool = False
    truncated_length: int | None = None
    # Optional activation data (collected separately)
    activations_path: str | None = None


@dataclass
class ConversationResult:
    """Result of a complete multi-turn conversation."""

    conversation_id: str
    condition: str
    model_name: str
    timestamp: str
    turns: list[TurnResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    @property
    def accuracy_by_turn(self) -> list[float]:
        """Get accuracy at each turn (cumulative)."""
        correct = 0
        accuracies = []
        for i, turn in enumerate(self.turns, 1):
            if turn.is_correct:
                correct += 1
            accuracies.append(correct / i)
        return accuracies

    @property
    def instant_accuracy_by_turn(self) -> list[int]:
        """Get binary correctness at each turn."""
        return [int(turn.is_correct) for turn in self.turns]

    @property
    def overall_accuracy(self) -> float:
        """Overall conversation accuracy."""
        if not self.turns:
            return 0.0
        return sum(t.is_correct for t in self.turns) / len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "condition": self.condition,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "total_time_seconds": self.total_time_seconds,
            "overall_accuracy": self.overall_accuracy,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "topic": t.topic,
                    "question_id": t.question_id,
                    "correct_answer": t.correct_answer,
                    "parsed_answer": t.parsed_answer,
                    "is_correct": t.is_correct,
                    "response_length": t.response_length,
                    "hedging_count": t.hedging_count,
                    "was_truncated": t.was_truncated,
                    "truncated_length": t.truncated_length,
                    "activations_path": t.activations_path,
                }
                for t in self.turns
            ],
        }


class ConversationRunner:
    """Runs multi-turn conversations with a model."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: AutoModelForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        device: str = "cuda",
    ):
        """Initialize conversation runner.

        Args:
            config: Experiment configuration
            model: Pre-loaded model (loads if None)
            tokenizer: Pre-loaded tokenizer (loads if None)
            device: Device to run on
        """
        self.config = config
        self.device = device

        # Load model and tokenizer if not provided
        if model is None or tokenizer is None:
            self.tokenizer, self.model = self._load_model()
        else:
            self.tokenizer = tokenizer
            self.model = model

        # Compile hedging pattern
        self._hedging_pattern = re.compile(
            "|".join(re.escape(h) for h in config.hedging_markers),
            re.IGNORECASE,
        )

    def _load_model(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load model and tokenizer."""
        model_id = self.config.model_config.hf_id

        print(f"Loading tokenizer from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"Loading model from {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        return tokenizer, model

    def count_hedging(self, text: str) -> int:
        """Count hedging markers in text."""
        return len(self._hedging_pattern.findall(text))

    def _truncate_response(self, response: str, max_tokens: int) -> str:
        """Truncate response to approximately max_tokens.

        For interrupt conditions, we truncate the response that goes into
        conversation history to simulate the model being cut off mid-response.
        """
        tokens = self.tokenizer.encode(response, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return response

        truncated_tokens = tokens[:max_tokens]
        truncated = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Add indicator that response was cut off (this is what the model sees)
        return truncated.rstrip() + "..."

    def _build_conversation_messages(
        self,
        history: list[dict[str, str]],
        new_question: Question,
        condition: Condition,
        is_first_turn: bool,
    ) -> list[dict[str, str]]:
        """Build message list for the model.

        Args:
            history: Previous messages
            new_question: Question for this turn
            condition: Experimental condition
            is_first_turn: Whether this is the first turn

        Returns:
            List of messages in chat format
        """
        messages = list(history)

        if is_first_turn:
            # First turn: just ask the question
            user_content = new_question.format_with_instruction()
        else:
            # Subsequent turns: apply condition template
            template = CONDITION_TEMPLATES[condition]
            question_text = new_question.format_with_instruction()
            user_content = template.format(topic=question_text)

        messages.append({"role": "user", "content": user_content})
        return messages

    def _generate_response(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate model response for messages.

        Args:
            messages: Conversation messages

        Returns:
            Model response text
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response.strip()

    def run_conversation(
        self,
        condition: Condition,
        questions: list[Question],
        conversation_id: str | None = None,
    ) -> ConversationResult:
        """Run a single multi-turn conversation.

        Args:
            condition: Experimental condition to use
            questions: Questions to ask (one per turn)
            conversation_id: Optional ID for this conversation

        Returns:
            ConversationResult with all turn data
        """
        import time

        if conversation_id is None:
            conversation_id = f"{condition.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = ConversationResult(
            conversation_id=conversation_id,
            condition=condition.value,
            model_name=self.config.model_config.name,
            timestamp=datetime.now().isoformat(),
        )

        history: list[dict[str, str]] = []
        start_time = time.time()

        for turn_num, question in enumerate(questions, 1):
            # Build messages
            messages = self._build_conversation_messages(
                history=history,
                new_question=question,
                condition=condition,
                is_first_turn=(turn_num == 1),
            )

            # Generate response
            response = self._generate_response(messages)

            # Check answer
            is_correct, parsed = check_answer(question, response)

            # For interrupt conditions, truncate response in conversation history
            # This simulates the model being cut off mid-response
            is_interrupt = condition in INTERRUPT_CONDITIONS
            is_last_turn = turn_num == len(questions)

            if is_interrupt and not is_last_turn:
                # Truncate for history - model sees its response was cut off
                history_response = self._truncate_response(
                    response,
                    self.config.interrupt_truncate_tokens,
                )
                was_truncated = True
                truncated_length = len(history_response)
            else:
                # Complete conditions: full response in history
                history_response = response
                was_truncated = False
                truncated_length = None

            # Create turn result (with full response for data)
            turn = TurnResult(
                turn_number=turn_num,
                topic=question.category,
                question_id=question.question_id,
                question_text=question.question,
                correct_answer=question.answer,
                model_response=response,
                parsed_answer=parsed,
                is_correct=is_correct,
                response_length=len(response),
                hedging_count=self.count_hedging(response),
                was_truncated=was_truncated,
                truncated_length=truncated_length,
            )
            result.turns.append(turn)

            # Update history for next turn
            history = messages + [{"role": "assistant", "content": history_response}]

        result.total_time_seconds = time.time() - start_time
        return result


def run_experiment_batch(
    config: ExperimentConfig,
    dataset: MMLUProDataset,
    topics: list[str],
    conditions: list[Condition] | None = None,
    progress_callback: Any = None,
) -> list[ConversationResult]:
    """Run a batch of conversations across conditions.

    Args:
        config: Experiment configuration
        dataset: MMLU-Pro dataset
        topics: List of topics to use
        conditions: Conditions to run (defaults to all)
        progress_callback: Optional callback for progress updates

    Returns:
        List of all conversation results
    """
    if conditions is None:
        conditions = list(Condition)

    # Initialize runner
    runner = ConversationRunner(config)

    results = []
    total_convos = len(conditions) * config.num_conversations_per_condition

    with tqdm(total=total_convos, desc="Running conversations") as pbar:
        for condition in conditions:
            for conv_idx in range(config.num_conversations_per_condition):
                # Get questions for this conversation
                questions = dataset.get_questions_for_conversation(
                    topics[:config.num_turns],
                    num_per_topic=1,
                )

                # Run conversation
                conv_id = f"{condition.value}_{conv_idx:03d}"
                result = runner.run_conversation(
                    condition=condition,
                    questions=questions,
                    conversation_id=conv_id,
                )

                results.append(result)
                pbar.update(1)

                # Progress callback
                if progress_callback:
                    progress_callback(result)

    return results


def save_results(
    results: list[ConversationResult],
    output_path: Path,
) -> None:
    """Save conversation results to JSON file.

    Args:
        results: List of conversation results
        output_path: Path to save to
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(results)} conversations to {output_path}")


def load_results(input_path: Path) -> list[dict[str, Any]]:
    """Load conversation results from JSON file.

    Args:
        input_path: Path to load from

    Returns:
        List of conversation data dictionaries
    """
    with open(input_path) as f:
        return json.load(f)
