from pathlib import Path

import pandas as pd

from tandem_rlvr.agents import OracleSeniorAgent, WeakJuniorAgent
from tandem_rlvr.eval import EvaluationRunner
from tandem_rlvr.tasks.base import Task


def test_eval_runner_computes_metrics_and_saves_csv(tmp_path: Path) -> None:
    tasks = [
        Task(
            task_id="easy",
            task_type="addition",
            prompt="What is 2 + 3?",
            answer="5",
            difficulty="easy",
            metadata={"left": 2, "right": 3, "operator": "+"},
        ),
        Task(
            task_id="hard",
            task_type="addition",
            prompt="What is 100 + 23?",
            answer="123",
            difficulty="hard",
            metadata={"left": 100, "right": 23, "operator": "+"},
        ),
    ]
    output_path = tmp_path / "results.csv"

    result = EvaluationRunner(OracleSeniorAgent(), WeakJuniorAgent()).run(tasks, output_path)

    assert result.summary.num_tasks == 2
    assert result.summary.senior_only_accuracy == 1.0
    assert result.summary.junior_only_accuracy == 0.5
    assert result.summary.tandem_handoff_accuracy == 1.0
    assert result.summary.handoff_gain == 0.5
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert len(saved) == 2
