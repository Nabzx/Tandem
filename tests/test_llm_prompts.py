from tandem_rlvr.agents.llm.prompts import junior_tandem_prompt, senior_handoff_prompt
from tandem_rlvr.tasks.base import Task


def test_senior_handoff_prompt_requests_no_final_answer() -> None:
    task = Task("t1", "addition", "What is 2 + 3?", "5", "easy", {"left": 2, "right": 3, "operator": "+"})

    prompt = senior_handoff_prompt(task)

    assert "Do not reveal the final answer" in prompt
    assert '"final_answer": null' in prompt
    assert task.prompt in prompt


def test_junior_tandem_prompt_includes_trust_warning_and_reasoning() -> None:
    task = Task("t1", "addition", "What is 2 + 3?", "5", "easy", {"left": 2, "right": 3, "operator": "+"})

    prompt = junior_tandem_prompt(task, "Add the two small numbers.")

    assert "do not blindly trust it" in prompt
    assert "Add the two small numbers." in prompt
