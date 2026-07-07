from __future__ import annotations

import random

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.tasks.base import Task


class RandomAgent:
    """A random baseline for sanity checks."""

    name = "random"

    def __init__(self, seed: int | None = None, answer_low: int = -100, answer_high: int = 100) -> None:
        self._rng = random.Random(seed)
        self.answer_low = answer_low
        self.answer_high = answer_high

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        guess = self._guess(task)
        return AgentResponse(
            reasoning="Guessed randomly.",
            final_answer=guess,
            metadata={"used_context": context is not None},
        )

    def _guess(self, task: Task) -> str:
        answer_type = task.metadata.get("answer_type", "int")
        if answer_type == "list_int":
            values = [self._rng.randint(-5, 10) for _ in range(self._rng.randint(0, 4))]
            return str(values)
        if answer_type == "bool":
            return self._rng.choice(["true", "false"])
        if answer_type == "text":
            return self._rng.choice(["Alice", "Bob", "Charlie", "unknown"])
        return str(self._rng.randint(self.answer_low, self.answer_high))
