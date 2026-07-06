from __future__ import annotations

import re

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.tasks.base import Task


_RESULT_RE = re.compile(r"(?:result|answer)\s+is\s+(-?\d+)", re.IGNORECASE)
_INTEGER_RE = re.compile(r"-?\d+")


class WeakJuniorAgent:
    """A limited arithmetic agent that benefits from simple senior handoffs."""

    name = "weak_junior"

    def __init__(self, direct_threshold: int = 20) -> None:
        self.direct_threshold = direct_threshold

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        if context:
            parsed = self._parse_handoff_result(context)
            if parsed is not None:
                return AgentResponse(
                    reasoning="Used the senior's handoff trace to identify the result.",
                    final_answer=parsed,
                    metadata={"mode": "handoff_parse"},
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
            final_answer="0",
            metadata={"mode": "failed_hard"},
        )

    @staticmethod
    def _parse_handoff_result(context: str) -> str | None:
        result_match = _RESULT_RE.search(context)
        if result_match:
            return result_match.group(1)
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
