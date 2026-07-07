from __future__ import annotations

import ast
import re
from typing import Any

from tandem_rlvr.tasks.base import Task


_INTEGER_RE = re.compile(r"-?\d+")
_BOOL_TRUE = {"true", "yes", "y"}
_BOOL_FALSE = {"false", "no", "n"}


def normalize_final_answer(answer: str | int | float | None) -> str:
    """Extract a canonical integer answer string when possible."""

    if answer is None:
        return ""
    text = str(answer).strip()
    if text == "":
        return ""
    matches = _INTEGER_RE.findall(text.replace(",", ""))
    if not matches:
        return text.lower()
    return str(int(matches[-1]))


def verify_final_answer(task: Task, candidate_answer: str | int | float | None) -> bool:
    answer_type = task.metadata.get("answer_type", "int")
    if answer_type == "list_int":
        return _normalize_int_list(candidate_answer) == _normalize_int_list(task.answer)
    if answer_type == "bool":
        return _normalize_bool(candidate_answer) == _normalize_bool(task.answer)
    if answer_type == "text":
        return _normalize_text(candidate_answer) == _normalize_text(task.answer)
    return normalize_final_answer(candidate_answer) == normalize_final_answer(task.answer)


def _normalize_int_list(answer: Any) -> list[int] | None:
    if isinstance(answer, list):
        try:
            return [int(value) for value in answer]
        except (TypeError, ValueError):
            return None

    text = str(answer).strip() if answer is not None else ""
    if text == "":
        return None

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, list):
        try:
            return [int(value) for value in parsed]
        except (TypeError, ValueError):
            return None

    if "[NUM]" in text:
        return None
    return [int(match) for match in _INTEGER_RE.findall(text.replace(",", ""))]


def _normalize_bool(answer: Any) -> bool | None:
    text = _normalize_text(answer)
    if text in _BOOL_TRUE:
        return True
    if text in _BOOL_FALSE:
        return False
    return None


def _normalize_text(answer: Any) -> str:
    if answer is None:
        return ""
    return re.sub(r"[^a-z0-9_ -]+", "", str(answer).strip().lower())
