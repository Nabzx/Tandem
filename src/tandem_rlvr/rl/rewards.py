from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RewardResult:
    total_reward: float
    correctness_reward: float
    process_reward_component: float
    usefulness_component: float
    leakage_penalty: float
    hallucination_penalty: float
    components: dict[str, Any]


def compute_handoff_reward(
    tandem_correct: bool,
    process_metrics: dict[str, Any],
) -> RewardResult:
    """Compute a first-pass heuristic RLVR reward for senior handoff strategies."""

    correctness_reward = 1.0 if tandem_correct else 0.0
    process_score = _float_or_zero(process_metrics.get("process_reward_score"))
    usefulness_score = _float_or_zero(process_metrics.get("usefulness_score"))
    leaks_exact_answer = bool(process_metrics.get("leaks_exact_answer", False))
    hallucination_flag = _has_hallucination(process_metrics.get("hallucination_flags"))

    process_reward_component = 0.30 * process_score
    usefulness_component = 0.20 * usefulness_score
    leakage_penalty = 0.50 if leaks_exact_answer else 0.0
    hallucination_penalty = 0.30 if hallucination_flag else 0.0
    total = correctness_reward + process_reward_component + usefulness_component - leakage_penalty - hallucination_penalty
    clipped = max(-1.0, min(2.0, total))

    return RewardResult(
        total_reward=clipped,
        correctness_reward=correctness_reward,
        process_reward_component=process_reward_component,
        usefulness_component=usefulness_component,
        leakage_penalty=leakage_penalty,
        hallucination_penalty=hallucination_penalty,
        components={
            "tandem_correct": tandem_correct,
            "process_reward_score": process_score,
            "usefulness_score": usefulness_score,
            "leaks_exact_answer": leaks_exact_answer,
            "hallucination_flag": hallucination_flag,
        },
    )


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_hallucination(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip() not in {"", "[]"}
    return bool(value)
