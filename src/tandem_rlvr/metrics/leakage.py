from __future__ import annotations

import ast
import re
from typing import Any

from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import normalize_final_answer


_INTEGER_RE = re.compile(r"-?\d+")


def compute_leakage_metrics(
    task: Task,
    reasoning: str | None,
    expected_answer: str | None = None,
) -> dict[str, Any]:
    """Detect whether a handoff trace gives away the final answer."""

    answer = task.answer if expected_answer is None else str(expected_answer)
    text = (reasoning or "").strip()
    text_lower = text.lower()
    answer_type = _infer_answer_type(task, answer)

    leaks_exact_answer = answer.strip() != "" and answer.strip() in text
    leaks_normalized_answer = False
    leaks_answer_digits = False

    if answer_type == "int":
        normalized = normalize_final_answer(answer)
        reasoning_numbers = [str(int(value)) for value in _INTEGER_RE.findall(text.replace(",", ""))]
        leaks_normalized_answer = normalized in reasoning_numbers
        leaks_answer_digits = normalized != "" and normalized in "".join(reasoning_numbers)
    elif answer_type == "list_int":
        normalized_answer = _normalize_list(answer)
        normalized_reasoning = _compact_list_text(text)
        leaks_normalized_answer = normalized_answer != "" and normalized_answer in normalized_reasoning
        leaks_answer_digits = bool(normalized_answer and all(num in text for num in _INTEGER_RE.findall(answer)))
    elif answer_type == "bool":
        normalized = answer.strip().lower()
        leaks_normalized_answer = normalized in {"true", "false"} and normalized in text_lower
        leaks_exact_answer = leaks_normalized_answer
    else:
        normalized = _normalize_text(answer)
        leaks_normalized_answer = normalized != "" and normalized in _normalize_text(text)

    leakage_score = 1.0
    if answer_type == "bool" and leaks_normalized_answer:
        leakage_score = 0.8
    elif leaks_exact_answer or leaks_normalized_answer:
        leakage_score = 0.2
    elif leaks_answer_digits:
        leakage_score = 0.5

    return {
        "leaks_exact_answer": leaks_exact_answer,
        "leaks_normalized_answer": leaks_normalized_answer,
        "leaks_answer_digits": leaks_answer_digits,
        "leakage_score": leakage_score,
    }


def _infer_answer_type(task: Task, answer: str) -> str:
    metadata_type = task.metadata.get("answer_type")
    if metadata_type:
        return str(metadata_type)
    stripped = answer.strip().lower()
    if stripped in {"true", "false"}:
        return "bool"
    if stripped.startswith("[") and stripped.endswith("]"):
        return "list_int"
    if normalize_final_answer(answer) == answer.strip() or _INTEGER_RE.fullmatch(answer.strip() or ""):
        return "int"
    return "text"


def _normalize_list(answer: str) -> str:
    try:
        parsed = ast.literal_eval(answer)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return "[" + ",".join(str(int(value)) for value in parsed) + "]"
    numbers = _INTEGER_RE.findall(answer)
    if numbers:
        return "[" + ",".join(str(int(value)) for value in numbers) + "]"
    return ""


def _compact_list_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
