from __future__ import annotations

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.tasks.base import Task


class OracleSeniorAgent:
    """A deterministic senior that solves generated synthetic tasks exactly."""

    name = "oracle_senior"

    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        if "left" in task.metadata and "right" in task.metadata and "operator" in task.metadata:
            reasoning = self._arithmetic_reasoning(task)
        elif task.task_type.startswith("list_"):
            reasoning = self._list_reasoning(task)
        elif task.task_type.startswith("logic_"):
            reasoning = self._logic_reasoning(task)
        elif task.task_type.startswith("code_trace_"):
            reasoning = self._code_reasoning(task)
        else:
            reasoning = f"Solve the task directly. The answer is {task.answer}."

        return AgentResponse(
            reasoning=reasoning,
            final_answer=task.answer,
            metadata={"used_context": context is not None},
        )

    def _arithmetic_reasoning(self, task: Task) -> str:
        left = int(task.metadata["left"])
        right = int(task.metadata["right"])
        symbol = task.metadata["operator"]
        return self._reasoning(left, right, symbol, task.answer)

    @staticmethod
    def _list_reasoning(task: Task) -> str:
        values = task.metadata["values"]
        operation = task.metadata["operation"]
        return f"Apply {operation} to {values}. The answer is {task.answer}."

    @staticmethod
    def _logic_reasoning(task: Task) -> str:
        return f"Follow the stated relations step by step. The answer is {task.answer}."

    @staticmethod
    def _code_reasoning(task: Task) -> str:
        return f"Trace each assignment using the generated code state. The answer is {task.answer}."

    @staticmethod
    def _reasoning(left: int, right: int, symbol: str, answer: str) -> str:
        if symbol == "+":
            return f"Compute {left} + {right}. The result is {answer}."
        if symbol == "-":
            return f"Compute {left} - {right}. The result is {answer}."
        if symbol == "*":
            return f"Compute {left} * {right}. The result is {answer}."
        return f"Compute the expression. The result is {answer}."
