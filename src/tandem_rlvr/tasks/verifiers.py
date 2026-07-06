from __future__ import annotations

import re

from tandem_rlvr.tasks.base import Task


_INTEGER_RE = re.compile(r"-?\d+")


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
    return normalize_final_answer(candidate_answer) == normalize_final_answer(task.answer)
