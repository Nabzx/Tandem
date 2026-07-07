import random
from pathlib import Path

from tandem_rlvr.agents import OracleSeniorAgent, WeakJuniorAgent
from tandem_rlvr.eval import EvaluationRunner
from tandem_rlvr.eval.perturbations import drop_numbers
from tandem_rlvr.tasks.base import Task


def test_corrupted_handoff_evaluation_adds_metrics(tmp_path: Path) -> None:
    tasks = [
        Task(
            task_id="hard-arith",
            task_type="addition",
            prompt="What is 100 + 23?",
            answer="123",
            difficulty="hard",
            metadata={"left": 100, "right": 23, "operator": "+"},
        ),
    ]

    runner = EvaluationRunner(
        OracleSeniorAgent(),
        WeakJuniorAgent(),
        perturbation_fn=lambda text, rng: drop_numbers(text, rng=random.Random(0), drop_prob=1.0),
        seed=0,
    )
    result = runner.run(tasks, tmp_path / "stage2_results.csv")

    assert result.summary.tandem_handoff_accuracy == 1.0
    assert result.summary.corrupted_handoff_accuracy == 0.0
    assert result.summary.robustness_drop == 1.0
    assert "corrupted_handoff_correct" in result.results.columns


def test_stage2_grouped_metrics_are_present() -> None:
    tasks = [
        Task(
            task_id="easy-list",
            task_type="list_reverse",
            prompt="Reverse [1, 2, 3].",
            answer="[3, 2, 1]",
            difficulty="easy",
            metadata={"answer_type": "list_int", "values": [1, 2, 3], "operation": "reverse"},
        )
    ]

    result = EvaluationRunner(OracleSeniorAgent(), WeakJuniorAgent(), seed=1).run(tasks)

    assert result.summary.task_type_counts == {"list_reverse": 1}
    assert "list_reverse" in result.summary.accuracy_by_task_type
    assert "easy" in result.summary.accuracy_by_difficulty
