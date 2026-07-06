from __future__ import annotations

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.tasks.base import Task


class OracleSeniorAgent:
    """A deterministic senior that solves generated arithmetic tasks exactly."""

    name = "oracle_senior"

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        left = int(task.metadata["left"])
        right = int(task.metadata["right"])
        symbol = task.metadata["operator"]
        answer = task.answer
        reasoning = self._reasoning(left, right, symbol, answer)
        return AgentResponse(
            reasoning=reasoning,
            final_answer=answer,
            metadata={"used_context": context is not None},
        )

    @staticmethod
    def _reasoning(left: int, right: int, symbol: str, answer: str) -> str:
        if symbol == "+":
            return f"Compute {left} + {right}. The result is {answer}."
        if symbol == "-":
            return f"Compute {left} - {right}. The result is {answer}."
        if symbol == "*":
            return f"Compute {left} * {right}. The result is {answer}."
        return f"Compute the expression. The result is {answer}."
