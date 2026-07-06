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
        guess = self._rng.randint(self.answer_low, self.answer_high)
        return AgentResponse(
            reasoning="Guessed randomly.",
            final_answer=str(guess),
            metadata={"used_context": context is not None},
        )
