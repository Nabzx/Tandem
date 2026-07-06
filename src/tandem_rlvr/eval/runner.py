from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tandem_rlvr.agents.base import Agent
from tandem_rlvr.eval.metrics import EvaluationSummary, compute_summary
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import verify_final_answer
from tandem_rlvr.utils.io import ensure_parent_dir


@dataclass(frozen=True)
class EvaluationResult:
    results: pd.DataFrame
    summary: EvaluationSummary


class EvaluationRunner:
    def __init__(self, senior_agent: Agent, junior_agent: Agent) -> None:
        self.senior_agent = senior_agent
        self.junior_agent = junior_agent

    def run(self, tasks: list[Task], output_path: str | Path | None = None) -> EvaluationResult:
        rows = [self._evaluate_task(task) for task in tasks]
        results = pd.DataFrame(rows)
        summary = compute_summary(results)

        if output_path is not None:
            output_path = Path(output_path)
            ensure_parent_dir(output_path)
            results.to_csv(output_path, index=False)

        return EvaluationResult(results=results, summary=summary)

    def _evaluate_task(self, task: Task) -> dict[str, object]:
        senior_only = self.senior_agent.answer(task)
        junior_only = self.junior_agent.answer(task)
        senior_handoff = self.senior_agent.answer(task)
        tandem = self.junior_agent.answer(task, context=senior_handoff.reasoning)

        senior_correct = verify_final_answer(task, senior_only.final_answer)
        junior_correct = verify_final_answer(task, junior_only.final_answer)
        tandem_correct = verify_final_answer(task, tandem.final_answer)

        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "difficulty": task.difficulty,
            "prompt": task.prompt,
            "answer": task.answer,
            "senior_only_answer": senior_only.final_answer,
            "senior_only_correct": senior_correct,
            "senior_reasoning": senior_handoff.reasoning,
            "junior_only_answer": junior_only.final_answer,
            "junior_only_correct": junior_correct,
            "tandem_handoff_answer": tandem.final_answer,
            "tandem_handoff_correct": tandem_correct,
            "tandem_junior_reasoning": tandem.reasoning,
        }
