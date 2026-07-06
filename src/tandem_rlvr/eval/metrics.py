from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EvaluationSummary:
    num_tasks: int
    senior_only_correct: int
    senior_only_incorrect: int
    junior_only_correct: int
    junior_only_incorrect: int
    tandem_handoff_correct: int
    tandem_handoff_incorrect: int
    senior_only_accuracy: float
    junior_only_accuracy: float
    tandem_handoff_accuracy: float
    handoff_gain: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "num_tasks": self.num_tasks,
            "senior_only_correct": self.senior_only_correct,
            "senior_only_incorrect": self.senior_only_incorrect,
            "junior_only_correct": self.junior_only_correct,
            "junior_only_incorrect": self.junior_only_incorrect,
            "tandem_handoff_correct": self.tandem_handoff_correct,
            "tandem_handoff_incorrect": self.tandem_handoff_incorrect,
            "senior_only_accuracy": self.senior_only_accuracy,
            "junior_only_accuracy": self.junior_only_accuracy,
            "tandem_handoff_accuracy": self.tandem_handoff_accuracy,
            "handoff_gain": self.handoff_gain,
        }


def compute_summary(results: pd.DataFrame) -> EvaluationSummary:
    num_tasks = len(results)
    senior_correct = int(results["senior_only_correct"].sum())
    junior_correct = int(results["junior_only_correct"].sum())
    tandem_correct = int(results["tandem_handoff_correct"].sum())

    senior_acc = _accuracy(senior_correct, num_tasks)
    junior_acc = _accuracy(junior_correct, num_tasks)
    tandem_acc = _accuracy(tandem_correct, num_tasks)

    return EvaluationSummary(
        num_tasks=num_tasks,
        senior_only_correct=senior_correct,
        senior_only_incorrect=num_tasks - senior_correct,
        junior_only_correct=junior_correct,
        junior_only_incorrect=num_tasks - junior_correct,
        tandem_handoff_correct=tandem_correct,
        tandem_handoff_incorrect=num_tasks - tandem_correct,
        senior_only_accuracy=senior_acc,
        junior_only_accuracy=junior_acc,
        tandem_handoff_accuracy=tandem_acc,
        handoff_gain=tandem_acc - junior_acc,
    )


def _accuracy(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return correct / total
