from __future__ import annotations

import re

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import verify_final_answer


_ANSWER_RE = re.compile(r"(?:result|answer)\s+is\s+(.+?)(?:\.|$)", re.IGNORECASE)
_INTEGER_RE = re.compile(r"-?\d+")


class WeakJuniorAgent:
    """A limited agent that benefits from clear senior handoffs."""

    name = "weak_junior"

    def __init__(self, direct_threshold: int = 20) -> None:
        self.direct_threshold = direct_threshold

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        if context:
            parsed = self._parse_handoff_result(context)
            if parsed is not None and verify_final_answer(task, parsed):
                return AgentResponse(
                    reasoning="Used the senior's handoff trace to identify the result.",
                    final_answer=parsed,
                    metadata={"mode": "handoff_parse"},
                )

        if task.difficulty == "easy":
            return AgentResponse(
                reasoning="Solved directly because this task is in my easy range.",
                final_answer=task.answer,
                metadata={"mode": "direct_easy"},
            )

        if not {"left", "right", "operator"}.issubset(task.metadata):
            return AgentResponse(
                reasoning="This task is beyond my reliable direct-solving range without a clean handoff.",
                final_answer=self._wrong_answer(task),
                metadata={"mode": "failed_hard"},
            )

        left = int(task.metadata["left"])
        right = int(task.metadata["right"])
        symbol = task.metadata["operator"]
        if abs(left) <= self.direct_threshold and abs(right) <= self.direct_threshold:
            answer = self._compute(left, right, symbol)
            return AgentResponse(
                reasoning="Solved directly because the numbers are small.",
                final_answer=str(answer),
                metadata={"mode": "direct_easy"},
            )

        return AgentResponse(
            reasoning="The numbers are beyond my reliable direct-solving range.",
            final_answer=self._wrong_answer(task),
            metadata={"mode": "failed_hard"},
        )

    @staticmethod
    def _parse_handoff_result(context: str) -> str | None:
        if "[NUM]" in context:
            return None
        answer_match = _ANSWER_RE.search(context)
        if answer_match:
            return answer_match.group(1).strip()
        numbers = _INTEGER_RE.findall(context)
        if len(numbers) >= 3:
            return numbers[-1]
        return None

    @staticmethod
    def _compute(left: int, right: int, symbol: str) -> int:
        if symbol == "+":
            return left + right
        if symbol == "-":
            return left - right
        if symbol == "*":
            return left * right
        raise ValueError(f"Unsupported operator: {symbol}")

    @staticmethod
    def _wrong_answer(task: Task) -> str:
        answer_type = task.metadata.get("answer_type", "int")
        if answer_type == "list_int":
            return "[]"
        if answer_type == "bool":
            return "false" if str(task.answer).lower() == "true" else "true"
        if answer_type == "text":
            return "unknown"
        return "0" if str(task.answer) != "0" else "1"
