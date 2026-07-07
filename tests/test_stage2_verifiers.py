from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import verify_final_answer


def test_list_verifier_accepts_python_list_and_text_numbers() -> None:
    task = Task(
        task_id="list-1",
        task_type="list_sort",
        prompt="Sort [3, 1, 2].",
        answer="[1, 2, 3]",
        difficulty="easy",
        metadata={"answer_type": "list_int"},
    )

    assert verify_final_answer(task, "[1, 2, 3]")
    assert verify_final_answer(task, "The answer is 1, 2, 3.")
    assert not verify_final_answer(task, "[1, 3, 2]")


def test_bool_and_text_verifiers_are_robust() -> None:
    bool_task = Task(
        task_id="logic-1",
        task_type="logic_implication",
        prompt="Is it guaranteed?",
        answer="true",
        difficulty="easy",
        metadata={"answer_type": "bool"},
    )
    text_task = Task(
        task_id="logic-2",
        task_type="logic_comparison",
        prompt="Who is shortest?",
        answer="Charlie",
        difficulty="easy",
        metadata={"answer_type": "text"},
    )

    assert verify_final_answer(bool_task, "Yes.")
    assert verify_final_answer(text_task, "charlie")
