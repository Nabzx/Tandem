from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

MODES = ("senior_only", "junior_only", "tandem_handoff", "corrupted_handoff")


@dataclass(frozen=True)
class EvaluationSummary:
    num_tasks: int
    senior_only_correct: int
    senior_only_incorrect: int
    junior_only_correct: int
    junior_only_incorrect: int
    tandem_handoff_correct: int
    tandem_handoff_incorrect: int
    corrupted_handoff_correct: int
    corrupted_handoff_incorrect: int
    senior_only_accuracy: float
    junior_only_accuracy: float
    tandem_handoff_accuracy: float
    corrupted_handoff_accuracy: float
    handoff_gain: float
    robustness_drop: float
    failure_counts_by_mode: dict[str, int]
    task_type_counts: dict[str, int]
    accuracy_by_task_type: dict[str, dict[str, float]]
    accuracy_by_difficulty: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_tasks": self.num_tasks,
            "senior_only_correct": self.senior_only_correct,
            "senior_only_incorrect": self.senior_only_incorrect,
            "junior_only_correct": self.junior_only_correct,
            "junior_only_incorrect": self.junior_only_incorrect,
            "tandem_handoff_correct": self.tandem_handoff_correct,
            "tandem_handoff_incorrect": self.tandem_handoff_incorrect,
            "corrupted_handoff_correct": self.corrupted_handoff_correct,
            "corrupted_handoff_incorrect": self.corrupted_handoff_incorrect,
            "senior_only_accuracy": self.senior_only_accuracy,
            "junior_only_accuracy": self.junior_only_accuracy,
            "tandem_handoff_accuracy": self.tandem_handoff_accuracy,
            "corrupted_handoff_accuracy": self.corrupted_handoff_accuracy,
            "handoff_gain": self.handoff_gain,
            "robustness_drop": self.robustness_drop,
            "failure_counts_by_mode": self.failure_counts_by_mode,
            "task_type_counts": self.task_type_counts,
            "accuracy_by_task_type": self.accuracy_by_task_type,
            "accuracy_by_difficulty": self.accuracy_by_difficulty,
        }


def compute_summary(results: pd.DataFrame) -> EvaluationSummary:
    num_tasks = len(results)
    senior_correct = int(results["senior_only_correct"].sum())
    junior_correct = int(results["junior_only_correct"].sum())
    tandem_correct = int(results["tandem_handoff_correct"].sum())
    corrupted_correct = int(results["corrupted_handoff_correct"].sum())

    senior_acc = _accuracy(senior_correct, num_tasks)
    junior_acc = _accuracy(junior_correct, num_tasks)
    tandem_acc = _accuracy(tandem_correct, num_tasks)
    corrupted_acc = _accuracy(corrupted_correct, num_tasks)

    return EvaluationSummary(
        num_tasks=num_tasks,
        senior_only_correct=senior_correct,
        senior_only_incorrect=num_tasks - senior_correct,
        junior_only_correct=junior_correct,
        junior_only_incorrect=num_tasks - junior_correct,
        tandem_handoff_correct=tandem_correct,
        tandem_handoff_incorrect=num_tasks - tandem_correct,
        corrupted_handoff_correct=corrupted_correct,
        corrupted_handoff_incorrect=num_tasks - corrupted_correct,
        senior_only_accuracy=senior_acc,
        junior_only_accuracy=junior_acc,
        tandem_handoff_accuracy=tandem_acc,
        corrupted_handoff_accuracy=corrupted_acc,
        handoff_gain=tandem_acc - junior_acc,
        robustness_drop=tandem_acc - corrupted_acc,
        failure_counts_by_mode=_failure_counts(results),
        task_type_counts=_task_type_counts(results),
        accuracy_by_task_type=_grouped_accuracy(results, "task_type"),
        accuracy_by_difficulty=_grouped_accuracy(results, "difficulty"),
    )


def _accuracy(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return correct / total


def _failure_counts(results: pd.DataFrame) -> dict[str, int]:
    return {
        mode: int((~results[f"{mode}_correct"].astype(bool)).sum())
        for mode in MODES
    }


def _task_type_counts(results: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in results["task_type"].value_counts().sort_index().items()}


def _grouped_accuracy(results: pd.DataFrame, group_column: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for group_value, group in results.groupby(group_column):
        grouped[str(group_value)] = {
            mode: _accuracy(int(group[f"{mode}_correct"].sum()), len(group))
            for mode in MODES
        }
    return grouped
