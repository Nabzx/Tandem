from __future__ import annotations

import random
from collections.abc import Sequence

from tandem_rlvr.tasks.base import Task


SUPPORTED_CODE_TASK_TYPES = (
    "code_trace_assignment",
    "code_trace_loop_sum",
)
SUPPORTED_CODE_DIFFICULTIES = ("easy", "medium", "hard", "mixed")


class CodeTracingTaskGenerator:
    """Generate safe code tracing tasks from fixed templates."""

    def __init__(
        self,
        seed: int | None = None,
        task_types: Sequence[str] = SUPPORTED_CODE_TASK_TYPES,
        difficulty: str = "mixed",
    ) -> None:
        unknown_types = set(task_types) - set(SUPPORTED_CODE_TASK_TYPES)
        if unknown_types:
            raise ValueError(f"Unsupported code task types: {sorted(unknown_types)}")
        if difficulty not in SUPPORTED_CODE_DIFFICULTIES:
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
        if task_type == "code_trace_assignment":
            return self._assignment_trace(difficulty)
        if task_type == "code_trace_loop_sum":
            return self._loop_sum(difficulty)
        raise ValueError(f"Unsupported code task type: {task_type}")

    def _choose_difficulty(self) -> str:
        if self.difficulty == "mixed":
            return self._rng.choice(("easy", "medium", "hard"))
        return self.difficulty

    def _assignment_trace(self, difficulty: str) -> Task:
        ranges = {"easy": (1, 5), "medium": (2, 12), "hard": (5, 30)}
        low, high = ranges[difficulty]
        start = self._rng.randint(low, high)
        addend = self._rng.randint(1, high)
        multiplier = self._rng.randint(2, 5)
        answer = (start + addend) * multiplier
        code = f"x = {start}\nx = x + {addend}\nx = x * {multiplier}"
        prompt = f"Trace this code:\n```python\n{code}\n```\nWhat is x?"
        return self._task("code_trace_assignment", prompt, answer, difficulty, {"start": start, "addend": addend, "multiplier": multiplier})

    def _loop_sum(self, difficulty: str) -> Task:
        specs = {"easy": (3, 1, 5), "medium": (5, 1, 12), "hard": (7, -10, 20)}
        length, low, high = specs[difficulty]
        items = [self._rng.randint(low, high) for _ in range(length)]
        answer = sum(items)
        code = f"items = {items}\ntotal = 0\nfor x in items:\n    total += x"
        prompt = f"Trace this code:\n```python\n{code}\n```\nWhat is total?"
        return self._task("code_trace_loop_sum", prompt, answer, difficulty, {"items": items})

    def _task(self, task_type: str, prompt: str, answer: int, difficulty: str, metadata: dict[str, object]) -> Task:
        return Task(
            task_id=f"code-{self._counter:06d}",
            task_type=task_type,
            prompt=prompt,
            answer=str(answer),
            difficulty=difficulty,
            metadata={**metadata, "answer_type": "int", "answer_value": answer},
        )
