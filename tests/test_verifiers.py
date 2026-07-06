from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import normalize_final_answer, verify_final_answer


def test_normalize_final_answer_extracts_last_integer() -> None:
    assert normalize_final_answer("The answer is 41.") == "41"
    assert normalize_final_answer("First 17, then -24") == "-24"


def test_verify_final_answer_accepts_text_with_answer() -> None:
    task = Task(
        task_id="t1",
        task_type="addition",
        prompt="What is 17 + 24?",
        answer="41",
        difficulty="medium",
    )

    assert verify_final_answer(task, "The answer is 41.")
    assert not verify_final_answer(task, "40")
