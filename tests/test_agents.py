from tandem_rlvr.agents import OracleSeniorAgent, RandomAgent, WeakJuniorAgent
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import verify_final_answer


def make_task(left: int, right: int, operator: str, answer: int, difficulty: str = "hard") -> Task:
    return Task(
        task_id="t1",
        task_type="arithmetic",
        prompt=f"What is {left} {operator} {right}?",
        answer=str(answer),
        difficulty=difficulty,
        metadata={"left": left, "right": right, "operator": operator},
    )


def test_oracle_senior_solves_task_and_returns_reasoning() -> None:
    task = make_task(17, 24, "+", 41)

    response = OracleSeniorAgent().answer(task)

    assert verify_final_answer(task, response.final_answer)
    assert "result is 41" in response.reasoning


def test_weak_junior_solves_easy_task_directly() -> None:
    task = make_task(7, 8, "+", 15, difficulty="easy")

    response = WeakJuniorAgent().answer(task)

    assert verify_final_answer(task, response.final_answer)
    assert response.metadata["mode"] == "direct_easy"


def test_weak_junior_uses_handoff_for_hard_task() -> None:
    task = make_task(100, 23, "+", 123)
    context = "Compute 100 + 23. The result is 123."

    response = WeakJuniorAgent().answer(task, context=context)

    assert verify_final_answer(task, response.final_answer)
    assert response.metadata["mode"] == "handoff_parse"


def test_random_agent_returns_answer_object() -> None:
    task = make_task(7, 8, "+", 15)

    response = RandomAgent(seed=0).answer(task)

    assert response.reasoning
    assert response.final_answer
