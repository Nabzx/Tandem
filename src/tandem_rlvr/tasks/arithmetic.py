from __future__ import annotations

import operator
import random
from collections.abc import Sequence

from tandem_rlvr.tasks.base import Task


SUPPORTED_TASK_TYPES = ("addition", "subtraction", "multiplication")
SUPPORTED_DIFFICULTIES = ("easy", "medium", "hard", "mixed")


class ArithmeticTaskGenerator:
    """Generate synthetic arithmetic tasks with exact integer answers."""

    def __init__(
        self,
        seed: int | None = None,
        task_types: Sequence[str] = SUPPORTED_TASK_TYPES,
        difficulty: str = "mixed",
    ) -> None:
        unknown_types = set(task_types) - set(SUPPORTED_TASK_TYPES)
        if unknown_types:
            raise ValueError(f"Unsupported task types: {sorted(unknown_types)}")
        if difficulty not in SUPPORTED_DIFFICULTIES:
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
        left, right = self._sample_operands(difficulty)
        answer = self._compute_answer(task_type, left, right)
        symbol = self._symbol(task_type)

        return Task(
            task_id=f"arith-{self._counter:06d}",
            task_type=task_type,
            prompt=f"What is {left} {symbol} {right}?",
            answer=str(answer),
            difficulty=difficulty,
            metadata={"left": left, "right": right, "operator": symbol},
        )

    def _choose_difficulty(self) -> str:
        if self.difficulty == "mixed":
            return self._rng.choice(("easy", "medium", "hard"))
        return self.difficulty

    def _sample_operands(self, difficulty: str) -> tuple[int, int]:
        ranges = {
            "easy": (0, 9),
            "medium": (10, 99),
            "hard": (100, 999),
        }
        low, high = ranges[difficulty]
        return self._rng.randint(low, high), self._rng.randint(low, high)

    @staticmethod
    def _compute_answer(task_type: str, left: int, right: int) -> int:
        ops = {
            "addition": operator.add,
            "subtraction": operator.sub,
            "multiplication": operator.mul,
        }
        return ops[task_type](left, right)

    @staticmethod
    def _symbol(task_type: str) -> str:
        return {
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
        }[task_type]
