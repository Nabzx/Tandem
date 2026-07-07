from tandem_rlvr.agents.llm.parsing import normalize_llm_answer, parse_llm_response
from tandem_rlvr.tasks.base import Task


def test_parse_strict_json() -> None:
    parsed = parse_llm_response('{"reasoning": "Add.", "final_answer": "5"}')

    assert parsed.reasoning == "Add."
    assert parsed.final_answer == "5"
    assert parsed.metadata["parse_status"] == "strict_json"


def test_parse_extracted_json() -> None:
    parsed = parse_llm_response('Sure: {"reasoning": "Add.", "final_answer": 5} Thanks.')

    assert parsed.final_answer == "5"
    assert parsed.metadata["parse_status"] == "extracted_json"


def test_parse_fallback_text() -> None:
    parsed = parse_llm_response("Reasoning here.\nFinal answer: true")

    assert parsed.final_answer == "true"
    assert parsed.metadata["parse_status"] == "fallback_text"


def test_normalize_llm_answer_common_cases() -> None:
    int_task = Task("t1", "addition", "What is 20 + 22?", "42", "easy", {})
    bool_task = Task("t2", "logic", "Is it true?", "true", "easy", {"answer_type": "bool"})
    text_task = Task("t3", "logic", "Who?", "Alice", "easy", {"answer_type": "text"})
    list_task = Task("t4", "list_sort", "Sort.", "[1, 2]", "easy", {"answer_type": "list_int"})

    assert normalize_llm_answer(int_task, "The answer is 42.") == "42"
    assert normalize_llm_answer(bool_task, "Yes.") == "true"
    assert normalize_llm_answer(text_task, "Alice!") == "alice"
    assert normalize_llm_answer(list_task, "[1, 2]") == "[1, 2]"
