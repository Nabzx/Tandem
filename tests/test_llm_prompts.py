from tandem_rlvr.agents.llm.prompts import junior_direct_prompt, junior_tandem_prompt, senior_direct_prompt, senior_handoff_prompt
from tandem_rlvr.tasks.base import Task


def test_direct_prompts_require_final_answer_and_do_not_discourage_answer() -> None:
    task = Task("t1", "addition", "What is 2 + 3?", "5", "easy", {"left": 2, "right": 3, "operator": "+"})

    senior_prompt = senior_direct_prompt(task)
    junior_prompt = junior_direct_prompt(task)

    for prompt in [senior_prompt, junior_prompt]:
        assert '"final_answer": "answer here"' in prompt
        assert "Do not reveal the final answer" not in prompt
        assert '{"reasoning": "Compute 7 * 8 = 56.", "final_answer": "56"}' in prompt
        assert "Keep reasoning to one short sentence." in prompt
        assert "The final_answer must be the shortest valid answer." in prompt
        assert "For arithmetic, compute directly and check the operation symbol carefully." in prompt


def test_senior_handoff_prompt_requests_no_final_answer() -> None:
    task = Task("t1", "addition", "What is 2 + 3?", "5", "easy", {"left": 2, "right": 3, "operator": "+"})

    prompt = senior_handoff_prompt(task)

    assert "Do not reveal the final answer" in prompt
    assert '"final_answer": null' in prompt
    assert task.prompt in prompt


def test_junior_tandem_prompt_requires_final_answer_and_includes_reasoning() -> None:
    task = Task("t1", "addition", "What is 2 + 3?", "5", "easy", {"left": 2, "right": 3, "operator": "+"})

    prompt = junior_tandem_prompt(task, "Add the two small numbers.")

    assert "do not blindly trust it" in prompt
    assert "Add the two small numbers." in prompt
    assert '"final_answer": "answer here"' in prompt
    assert "Do not reveal the final answer" not in prompt


def test_task_specific_prompt_instructions() -> None:
    list_task = Task("list-1", "list_sort", "Sort [3, 1, 2].", "[1, 2, 3]", "easy", {"answer_type": "list_int"})
    logic_task = Task("logic-1", "logic_implication", "Is it guaranteed?", "true", "easy", {"answer_type": "bool"})

    list_prompt = junior_direct_prompt(list_task)
    logic_prompt = senior_direct_prompt(logic_task)

    assert 'final_answer must be a complete Python-style list, e.g. "[1, 2, 3]"' in list_prompt
    assert 'final_answer must be exactly "true" or "false"' in logic_prompt
    assert '{"reasoning": "Add 4 to each list element.", "final_answer": "[13, 20, 31]"}' in list_prompt
    assert '{"reasoning": "The implication is transitive.", "final_answer": "true"}' in logic_prompt
