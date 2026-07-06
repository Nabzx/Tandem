from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tandem_rlvr.tasks.base import Task


@dataclass(frozen=True)
class AgentResponse:
    reasoning: str
    final_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    name: str

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        """Answer a task, optionally using handoff context."""
