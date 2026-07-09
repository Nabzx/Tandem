from __future__ import annotations

import re
from typing import Any

from tandem_rlvr.tasks.base import Task


_NUMBER_RE = re.compile(r"-?\d+")
_SENTENCE_RE = re.compile(r"[.!?]+")


def compute_legibility_metrics(
    task: Task,
    reasoning: str | None,
    final_answer: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute simple deterministic readability features for a reasoning trace."""

    text = (reasoning or "").strip()
    words = text.split()
    sentences = [sentence.strip() for sentence in _SENTENCE_RE.split(text) if sentence.strip()]
    sentence_count = len(sentences)
    word_count = len(words)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
    too_short = word_count < 4
    too_long = word_count > 80
    contains_operation_hint = _contains_operation_hint(task, text)
    has_step_markers = _has_step_markers(text)

    score = 1.0
    if not text:
        score -= 0.7
    if too_short:
        score -= 0.25
    if too_long:
        score -= 0.25
    if sentence_count > 6:
        score -= 0.15
    if contains_operation_hint:
        score += 0.08
    if has_step_markers:
        score += 0.05

    return {
        "reasoning_word_count": word_count,
        "reasoning_char_count": len(text),
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "too_short": too_short,
        "too_long": too_long,
        "contains_number": bool(_NUMBER_RE.search(text)),
        "contains_operation_hint": contains_operation_hint,
        "has_step_markers": has_step_markers,
        "legibility_score": _clamp(score),
    }


def _contains_operation_hint(task: Task, reasoning: str) -> bool:
    text = reasoning.lower()
    task_type = task.task_type
    keywords_by_family = {
        "addition": ["+", "add", "sum", "plus"],
        "subtraction": ["-", "subtract", "minus", "difference"],
        "multiplication": ["*", "multiply", "product", "times"],
        "list_sort": ["sort", "ascending"],
        "list_filter_even": ["even", "filter"],
        "list_filter_odd": ["odd", "filter"],
        "list_map_add": ["add", "each"],
        "list_map_multiply": ["multiply", "each"],
        "list_reverse": ["reverse"],
        "logic_syllogism": ["all", "implies", "therefore", "transitive"],
        "logic_comparison": ["taller", "shortest", "compare"],
        "logic_implication": ["if", "then", "implies", "guaranteed"],
        "code_trace_assignment": ["x", "assignment", "trace"],
        "code_trace_loop_sum": ["loop", "sum", "total"],
    }
    return any(keyword in text for keyword in keywords_by_family.get(task_type, []))


def _has_step_markers(reasoning: str) -> bool:
    text = reasoning.lower()
    return any(marker in text for marker in ["first", "then", "next", "therefore", "so ", "step"])


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
