from __future__ import annotations

from typing import Any

import pandas as pd


PROCESS_WEIGHTS = {
    "legibility_score": 0.30,
    "leakage_score": 0.25,
    "relevance_score": 0.25,
    "usefulness_score": 0.20,
}


def compute_usefulness_metrics_by_task(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for task_id, group in results.groupby("task_id"):
        junior = _mode_correct(group, "junior_only")
        tandem = _mode_correct(group, "tandem_handoff")
        corrupted = _mode_correct(group, "corrupted_handoff")
        metrics[str(task_id)] = compute_usefulness_metrics(junior, tandem, corrupted)
    return metrics


def compute_usefulness_metrics(
    junior_correct_without_handoff: bool | None,
    junior_correct_with_handoff: bool | None,
    junior_correct_with_corrupted_handoff: bool | None,
) -> dict[str, Any]:
    handoff_improved = (
        junior_correct_without_handoff is False and junior_correct_with_handoff is True
    )
    handoff_hurt = (
        junior_correct_without_handoff is True and junior_correct_with_handoff is False
    )
    corruption_hurt = (
        junior_correct_with_handoff is True and junior_correct_with_corrupted_handoff is False
    )

    score: float | None
    if junior_correct_with_handoff is None:
        score = None
    elif handoff_improved:
        score = 1.0
    elif handoff_hurt:
        score = 0.0
    elif junior_correct_without_handoff is True and junior_correct_with_handoff is True:
        score = 0.75
    elif junior_correct_without_handoff is False and junior_correct_with_handoff is False:
        score = 0.35
    else:
        score = 0.5

    return {
        "junior_correct_without_handoff": junior_correct_without_handoff,
        "junior_correct_with_handoff": junior_correct_with_handoff,
        "junior_correct_with_corrupted_handoff": junior_correct_with_corrupted_handoff,
        "handoff_improved": handoff_improved,
        "handoff_hurt": handoff_hurt,
        "corruption_hurt": corruption_hurt,
        "usefulness_score": score,
    }


def compute_process_reward(components: dict[str, Any]) -> dict[str, Any]:
    available = [
        key
        for key in PROCESS_WEIGHTS
        if components.get(key) is not None
    ]
    if not available:
        return {
            "process_reward_score": None,
            "process_reward_components_available": [],
        }

    total_weight = sum(PROCESS_WEIGHTS[key] for key in available)
    weighted_score = sum(float(components[key]) * PROCESS_WEIGHTS[key] for key in available)
    return {
        "process_reward_score": weighted_score / total_weight,
        "process_reward_components_available": available,
    }


def _mode_correct(group: pd.DataFrame, mode: str) -> bool | None:
    rows = group[group["mode"] == mode]
    if rows.empty:
        return None
    return bool(rows.iloc[0]["correct"])
