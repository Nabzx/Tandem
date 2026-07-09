from pathlib import Path

import pandas as pd

from tandem_rlvr.experiments.run_stage4_process_metrics import run_stage4_process_metrics
from tandem_rlvr.metrics import (
    compute_legibility_metrics,
    compute_leakage_metrics,
    compute_process_reward,
    compute_relevance_metrics,
    compute_usefulness_metrics_by_task,
)
from tandem_rlvr.tasks.base import Task


def test_legibility_metrics_empty_short_and_long_reasoning() -> None:
    task = Task("t1", "addition", "What is 2 + 3?", "5", "easy", {})

    empty = compute_legibility_metrics(task, "")
    short = compute_legibility_metrics(task, "Add.")
    long = compute_legibility_metrics(task, " ".join(["add"] * 90))

    assert empty["too_short"]
    assert short["too_short"]
    assert long["too_long"]
    assert empty["legibility_score"] < short["legibility_score"] < 1.0


def test_integer_answer_leakage_detection() -> None:
    task = Task("t1", "multiplication", "What is 7 * 8?", "56", "easy", {})

    leaked = compute_leakage_metrics(task, "Compute directly. The answer is 56.")
    clean = compute_leakage_metrics(task, "Compute the product carefully.")

    assert leaked["leaks_normalized_answer"]
    assert leaked["leakage_score"] < clean["leakage_score"]


def test_list_and_boolean_leakage_detection() -> None:
    list_task = Task("l1", "list_sort", "Sort [3, 1, 2].", "[1, 2, 3]", "easy", {"answer_type": "list_int"})
    bool_task = Task("b1", "logic_implication", "Is it guaranteed?", "true", "easy", {"answer_type": "bool"})

    list_metrics = compute_leakage_metrics(list_task, "The final list is [1, 2, 3].")
    bool_metrics = compute_leakage_metrics(bool_task, "The statement is true.")

    assert list_metrics["leaks_normalized_answer"]
    assert bool_metrics["leaks_normalized_answer"]
    assert bool_metrics["leakage_score"] == 0.8


def test_relevance_and_hallucination_metrics() -> None:
    arithmetic = Task("a1", "addition", "What is 2 + 3?", "5", "easy", {})
    list_task = Task("l1", "list_filter_even", "Return evens from [1, 2].", "[2]", "easy", {})
    logic = Task("g1", "logic_syllogism", "If all A are B and all B are C, are all A C?", "true", "easy", {})
    code = Task("c1", "code_trace_loop_sum", "Trace total in a loop.", "6", "easy", {})

    assert compute_relevance_metrics(arithmetic, "Add 2 plus 3.")["mentions_operation"]
    assert compute_relevance_metrics(list_task, "Filter the even values from the list.")["mentions_operation"]
    assert compute_relevance_metrics(logic, "This follows by transitive implication.")["mentions_operation"]
    assert compute_relevance_metrics(code, "Trace the loop and total sum.")["mentions_operation"]

    suspicious = compute_relevance_metrics(arithmetic, "Add them but do not go over 100 and divide by 1.")
    assert "go over 100" in suspicious["hallucination_flags"]
    assert "divide by 1" in suspicious["hallucination_flags"]


def test_usefulness_metrics_compare_junior_and_handoff() -> None:
    rows = pd.DataFrame(
        [
            {"task_id": "t1", "mode": "junior_only", "correct": False},
            {"task_id": "t1", "mode": "tandem_handoff", "correct": True},
            {"task_id": "t1", "mode": "corrupted_handoff", "correct": False},
        ]
    )

    metrics = compute_usefulness_metrics_by_task(rows)["t1"]

    assert metrics["handoff_improved"]
    assert metrics["corruption_hurt"]
    assert metrics["usefulness_score"] == 1.0


def test_process_reward_score_uses_available_components() -> None:
    reward = compute_process_reward(
        {
            "legibility_score": 1.0,
            "leakage_score": 0.5,
            "relevance_score": 1.0,
            "usefulness_score": None,
        }
    )

    assert reward["process_reward_score"] is not None
    assert reward["process_reward_components_available"] == [
        "legibility_score",
        "leakage_score",
        "relevance_score",
    ]


def test_stage4_experiment_on_tiny_stage3_csv(tmp_path: Path) -> None:
    stage3 = pd.DataFrame(
        [
            {
                "task_id": "t1",
                "task_type": "addition",
                "difficulty": "easy",
                "mode": "junior_only",
                "prompt": "What is 2 + 3?",
                "raw_expected_answer": "5",
                "expected_answer": "5",
                "raw_model_answer": "4",
                "model_answer": "4",
                "correct": False,
                "senior_handoff_reasoning": "",
                "senior_reasoning": "",
                "corrupted_reasoning": "",
            },
            {
                "task_id": "t1",
                "task_type": "addition",
                "difficulty": "easy",
                "mode": "tandem_handoff",
                "prompt": "What is 2 + 3?",
                "raw_expected_answer": "5",
                "expected_answer": "5",
                "raw_model_answer": "5",
                "model_answer": "5",
                "correct": True,
                "senior_handoff_reasoning": "Add the two numbers carefully.",
                "senior_handoff_raw_output": '{"reasoning": "Add carefully.", "final_answer": null}',
                "senior_handoff_parse_status": "strict_json",
                "corrupted_reasoning": "",
            },
            {
                "task_id": "t1",
                "task_type": "addition",
                "difficulty": "easy",
                "mode": "corrupted_handoff",
                "prompt": "What is 2 + 3?",
                "raw_expected_answer": "5",
                "expected_answer": "5",
                "raw_model_answer": "4",
                "model_answer": "4",
                "correct": False,
                "senior_handoff_reasoning": "Add the two numbers carefully.",
                "senior_handoff_raw_output": '{"reasoning": "Add carefully.", "final_answer": null}',
                "senior_handoff_parse_status": "strict_json",
                "corrupted_reasoning": "Add the numbers but go over 100.",
            },
        ]
    )
    input_path = tmp_path / "stage3.csv"
    stage3.to_csv(input_path, index=False)

    result = run_stage4_process_metrics(input_path, tmp_path)

    assert result["metrics_path"].exists()
    assert result["summary_path"].exists()
    assert result["summary"]["num_rows_scored"] == 2
    assert result["summary"]["num_tasks"] == 1
    assert "task_type" in result["summary"]["breakdowns"]
