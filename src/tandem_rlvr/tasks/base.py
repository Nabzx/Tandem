from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """A verifiable reasoning task."""

    task_id: str
    task_type: str
    prompt: str
    answer: str
    difficulty: str
    metadata: dict[str, Any] = field(default_factory=dict)
