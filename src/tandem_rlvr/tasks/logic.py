from __future__ import annotations

import random
from collections.abc import Sequence

from tandem_rlvr.tasks.base import Task


SUPPORTED_LOGIC_TASK_TYPES = (
    "logic_syllogism",
    "logic_comparison",
    "logic_implication",
)
SUPPORTED_LOGIC_DIFFICULTIES = ("easy", "medium", "hard", "mixed")


class LogicTaskGenerator:
    """Generate tiny symbolic logic tasks with known answers."""

    def __init__(
        self,
        seed: int | None = None,
        task_types: Sequence[str] = SUPPORTED_LOGIC_TASK_TYPES,
        difficulty: str = "mixed",
    ) -> None:
        unknown_types = set(task_types) - set(SUPPORTED_LOGIC_TASK_TYPES)
        if unknown_types:
            raise ValueError(f"Unsupported logic task types: {sorted(unknown_types)}")
        if difficulty not in SUPPORTED_LOGIC_DIFFICULTIES:
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
        if task_type == "logic_syllogism":
            return self._syllogism(difficulty)
        if task_type == "logic_comparison":
            return self._comparison(difficulty)
        if task_type == "logic_implication":
            return self._implication(difficulty)
        raise ValueError(f"Unsupported logic task type: {task_type}")

    def _choose_difficulty(self) -> str:
        if self.difficulty == "mixed":
            return self._rng.choice(("easy", "medium", "hard"))
        return self.difficulty

    def _syllogism(self, difficulty: str) -> Task:
        nouns = self._rng.sample(["blickets", "daxes", "wugs", "lorps", "mips", "tavs"], 3)
        answer = self._rng.choice([True, False]) if difficulty == "hard" else True
        if answer:
            prompt = f"If all {nouns[0]} are {nouns[1]}, and all {nouns[1]} are {nouns[2]}, are all {nouns[0]} {nouns[2]}?"
        else:
            prompt = f"If all {nouns[0]} are {nouns[1]}, and all {nouns[2]} are {nouns[1]}, are all {nouns[0]} {nouns[2]}?"
        return self._task("logic_syllogism", prompt, answer, difficulty, {"terms": nouns})

    def _comparison(self, difficulty: str) -> Task:
        names = self._rng.sample(["Alice", "Bob", "Charlie", "Dana", "Eli"], 3 if difficulty != "hard" else 4)
        prompt_parts = [f"{names[index]} is taller than {names[index + 1]}." for index in range(len(names) - 1)]
        prompt = " ".join(prompt_parts) + " Who is shortest?"
        answer = names[-1]
        return self._task("logic_comparison", prompt, answer, difficulty, {"ordered_tallest_to_shortest": names})

    def _implication(self, difficulty: str) -> Task:
        condition = self._rng.choice(["it rains", "the switch is on", "the token is blue"])
        consequence = self._rng.choice(["the ground is wet", "the lamp is lit", "the gate opens"])
        observed_condition = self._rng.choice([True, False]) if difficulty != "easy" else True
        if observed_condition:
            prompt = f"If {condition}, then {consequence}. {condition.capitalize()}. Is it guaranteed that {consequence}?"
            answer = True
        else:
            prompt = f"If {condition}, then {consequence}. {consequence.capitalize()}. Is it guaranteed that {condition}?"
            answer = False
        return self._task("logic_implication", prompt, answer, difficulty, {"condition": condition, "consequence": consequence})

    def _task(self, task_type: str, prompt: str, answer: bool | str, difficulty: str, metadata: dict[str, object]) -> Task:
        answer_type = "bool" if isinstance(answer, bool) else "text"
        answer_text = "true" if answer is True else "false" if answer is False else str(answer)
        return Task(
            task_id=f"logic-{self._counter:06d}",
            task_type=task_type,
            prompt=prompt,
            answer=answer_text,
            difficulty=difficulty,
            metadata={**metadata, "answer_type": answer_type, "answer_value": answer},
        )
