from __future__ import annotations

import random
from collections.abc import Sequence

from tandem_rlvr.tasks.base import Task


SUPPORTED_LIST_TASK_TYPES = (
    "list_sort",
    "list_filter_even",
    "list_filter_odd",
    "list_map_add",
    "list_map_multiply",
    "list_reverse",
)
SUPPORTED_LIST_DIFFICULTIES = ("easy", "medium", "hard", "mixed")


class ListTransformationTaskGenerator:
    """Generate small, verifiable list transformation tasks."""

    def __init__(
        self,
        seed: int | None = None,
        task_types: Sequence[str] = SUPPORTED_LIST_TASK_TYPES,
        difficulty: str = "mixed",
    ) -> None:
        unknown_types = set(task_types) - set(SUPPORTED_LIST_TASK_TYPES)
        if unknown_types:
            raise ValueError(f"Unsupported list task types: {sorted(unknown_types)}")
        if difficulty not in SUPPORTED_LIST_DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {difficulty}")

        self._rng = random.Random(seed)
        self.task_types = tuple(task_types)
        self.difficulty = difficulty
        self._counter = 0

    def generate(self, n: int) -> list[Task]:
        if n < 0:
            raise ValueError("n must be non-negative")
        return [self.generate_one() for _ in range(n)]

    def generate_one(self) -> Task:
        self._counter += 1
        task_type = self._rng.choice(self.task_types)
        difficulty = self._choose_difficulty()
        values = self._sample_values(difficulty)
        answer, prompt, operation = self._build_task(task_type, values)

        return Task(
            task_id=f"list-{self._counter:06d}",
            task_type=task_type,
            prompt=prompt,
            answer=str(answer),
            difficulty=difficulty,
            metadata={
                "values": values,
                "operation": operation,
                "answer_type": "list_int",
                "answer_value": answer,
            },
        )

    def _choose_difficulty(self) -> str:
        if self.difficulty == "mixed":
            return self._rng.choice(("easy", "medium", "hard"))
        return self.difficulty

    def _sample_values(self, difficulty: str) -> list[int]:
        specs = {
            "easy": (3, 0, 9),
            "medium": (5, 0, 30),
            "hard": (8, -25, 50),
        }
        length, low, high = specs[difficulty]
        return [self._rng.randint(low, high) for _ in range(length)]

    def _build_task(self, task_type: str, values: list[int]) -> tuple[list[int], str, str]:
        if task_type == "list_sort":
            return sorted(values), f"Given the list {values}, sort it in ascending order.", "sort"
        if task_type == "list_filter_even":
            return [value for value in values if value % 2 == 0], f"Given the list {values}, return only the even numbers.", "filter_even"
        if task_type == "list_filter_odd":
            return [value for value in values if value % 2 != 0], f"Given the list {values}, return only the odd numbers.", "filter_odd"
        if task_type == "list_map_add":
            increment = self._rng.randint(1, 5)
            answer = [value + increment for value in values]
            return answer, f"Given the list {values}, add {increment} to each element.", f"map_add:{increment}"
        if task_type == "list_map_multiply":
            factor = self._rng.randint(2, 4)
            answer = [value * factor for value in values]
            return answer, f"Given the list {values}, multiply each element by {factor}.", f"map_multiply:{factor}"
        if task_type == "list_reverse":
            return list(reversed(values)), f"Given the list {values}, reverse the list.", "reverse"
        raise ValueError(f"Unsupported list task type: {task_type}")
