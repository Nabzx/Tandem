import argparse
from pathlib import Path

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.agents.llm import OllamaBackendUnavailable
from tandem_rlvr.experiments.run_stage3_llm_eval import build_stage3_benchmark, main, parse_modes, parse_task_families, resolve_generation_settings, run_stage3_llm_eval
from tandem_rlvr.tasks.base import Task


class FakeSenior:
    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        return _response(task.answer)

    def produce_handoff(self, task: Task) -> AgentResponse:
        return AgentResponse(
            reasoning=f"Helpful partial reasoning. The answer is {task.answer}.",
            final_answer="",
            metadata={"raw_output": '{"reasoning": "handoff", "final_answer": null}', "parse_status": "strict_json"},
        )


class FakeJunior:
    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        if context and "[NUM]" not in context:
            return _response(task.answer)
        return _response(task.answer if task.difficulty == "easy" else "wrong")


class MissingAnswerSenior:
    def answer(self, task: Task, context: str | None = None) -> AgentResponse:
        return AgentResponse(
            reasoning="I forgot the answer field.",
            final_answer="",
            metadata={"raw_output": '{"reasoning": "I forgot the answer field."}', "parse_status": "strict_json"},
        )


def test_stage3_eval_smoke_with_fake_agents(tmp_path: Path) -> None:
    tasks = [
        Task("t1", "addition", "What is 2 + 3?", "5", "easy", {}),
        Task("t2", "addition", "What is 100 + 23?", "123", "hard", {}),
    ]

    result = run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        senior_model="fake-senior",
        junior_model="fake-junior",
        seed=0,
        results_path=tmp_path / "stage3_llm_results.csv",
        summary_path=tmp_path / "stage3_llm_summary.json",
        debug_prompts_dir=tmp_path / "debug_prompts",
    )

    assert len(result["results"]) == 8
    assert "senior_handoff" not in set(result["results"]["mode"])
    assert result["summary"]["total_tasks"] == 2
    assert result["summary"]["senior_model"] == "fake-senior"
    assert "parse_status_counts_by_mode" in result["summary"]
    assert "failure_type_counts_by_mode" in result["summary"]
    assert "accuracy_by_mode_and_task_type" in result["summary"]
    assert "normalized_expected_answer" in result["results"].columns
    assert "normalized_model_answer" in result["results"].columns
    assert (tmp_path / "stage3_llm_results.csv").exists()
    assert (tmp_path / "stage3_llm_summary.json").exists()
    assert any((tmp_path / "debug_prompts").glob("*_senior_only_fake-senior.txt"))


def test_tandem_rows_include_internal_senior_handoff_metadata(tmp_path: Path) -> None:
    tasks = [Task("t1", "addition", "What is 2 + 3?", "5", "easy", {})]

    result = run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        senior_model="fake-senior",
        junior_model="fake-junior",
        seed=0,
        results_path=tmp_path / "stage3_llm_results.csv",
        summary_path=tmp_path / "stage3_llm_summary.json",
        modes=parse_modes("tandem_handoff"),
    )

    row = result["results"].iloc[0]
    assert row["mode"] == "tandem_handoff"
    assert row["senior_handoff_reasoning"] == "Helpful partial reasoning. The answer is 5."
    assert row["senior_handoff_raw_output"] == '{"reasoning": "handoff", "final_answer": null}'
    assert row["senior_handoff_parse_status"] == "strict_json"
    assert row["senior_handoff_generation_seconds"] is not None
    assert result["summary"]["junior_only_accuracy"] is None
    assert result["summary"]["handoff_gain"] is None
    assert result["summary"]["robustness_drop"] is None


def test_corrupted_rows_include_corrupted_reasoning_metadata(tmp_path: Path) -> None:
    tasks = [Task("t1", "addition", "What is 100 + 23?", "123", "hard", {})]

    result = run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        senior_model="fake-senior",
        junior_model="fake-junior",
        seed=0,
        results_path=tmp_path / "stage3_llm_results.csv",
        summary_path=tmp_path / "stage3_llm_summary.json",
        modes=parse_modes("corrupted"),
    )

    row = result["results"].iloc[0]
    assert row["mode"] == "corrupted_handoff"
    assert row["senior_handoff_reasoning"]
    assert row["corrupted_reasoning"]
    assert result["summary"]["corrupted_handoff_accuracy"] is not None
    assert result["summary"]["tandem_handoff_accuracy"] is None
    assert result["summary"]["robustness_drop"] is None


def test_num_predict_explicit_override_beats_quick() -> None:
    args = argparse.Namespace(
        num_predict=192,
        quick=True,
        timeout_seconds=None,
        max_generation_seconds=None,
    )

    num_predict, timeout_seconds = resolve_generation_settings(args)

    assert num_predict == 192
    assert timeout_seconds == 30


def test_quick_and_normal_generation_defaults() -> None:
    quick_args = argparse.Namespace(num_predict=None, quick=True, timeout_seconds=None, max_generation_seconds=None)
    normal_args = argparse.Namespace(num_predict=None, quick=False, timeout_seconds=None, max_generation_seconds=None)

    assert resolve_generation_settings(quick_args) == (128, 30)
    assert resolve_generation_settings(normal_args) == (256, 60)


def test_missing_final_answer_is_empty_answer_failure(tmp_path: Path) -> None:
    tasks = [Task("t1", "addition", "What is 2 + 3?", "5", "easy", {})]

    result = run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=MissingAnswerSenior(),
        junior_agent=FakeJunior(),
        senior_model="fake-senior",
        junior_model="fake-junior",
        seed=0,
        results_path=tmp_path / "stage3_llm_results.csv",
        summary_path=tmp_path / "stage3_llm_summary.json",
        modes=parse_modes("senior_only"),
    )

    row = result["results"].iloc[0]
    assert row["parse_status"] == "strict_json"
    assert row["raw_model_answer"] == ""
    assert row["normalized_model_answer"] == ""
    assert not bool(row["correct"])
    assert row["failure_type"] == "empty_answer"
    assert result["summary"]["failure_type_counts_by_mode"]["senior_only"]["empty_answer"] == 1


def test_stage3_eval_can_run_single_mode(tmp_path: Path, capsys) -> None:
    tasks = [
        Task("t1", "addition", "What is 2 + 3?", "5", "easy", {}),
        Task("t2", "addition", "What is 100 + 23?", "123", "hard", {}),
    ]

    result = run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        senior_model="fake-senior",
        junior_model="fake-junior",
        seed=0,
        results_path=tmp_path / "stage3_llm_results.csv",
        summary_path=tmp_path / "stage3_llm_summary.json",
        modes=parse_modes("senior_only"),
        verbose=True,
    )

    assert list(result["results"]["mode"].unique()) == ["senior_only"]
    assert len(result["results"]) == 2
    assert result["summary"]["junior_only_accuracy"] is None
    assert result["summary"]["tandem_handoff_accuracy"] is None
    assert result["summary"]["corrupted_handoff_accuracy"] is None
    assert result["summary"]["handoff_gain"] is None
    assert result["summary"]["robustness_drop"] is None
    output = capsys.readouterr().out
    assert "mode=senior_only" in output
    assert "prompt preview:" in output
    assert "parse status:" in output


def test_stage3_task_family_filter_and_easy_only() -> None:
    tasks = build_stage3_benchmark(
        num_tasks=4,
        seed=42,
        difficulty="easy",
        task_families=parse_task_families("arithmetic"),
    )

    assert len(tasks) == 4
    assert all(task.task_type in {"addition", "subtraction", "multiplication"} for task in tasks)
    assert all(task.difficulty == "easy" for task in tasks)


def test_stage3_main_exits_cleanly_when_ollama_unavailable(monkeypatch, capsys) -> None:
    def raise_unavailable(*args, **kwargs):
        raise OllamaBackendUnavailable("Ollama backend unavailable. Please install Ollama.")

    monkeypatch.setattr("tandem_rlvr.experiments.run_stage3_llm_eval.run_llm_eval_from_args", raise_unavailable)
    monkeypatch.setattr("sys.argv", ["run_stage3_llm_eval", "--num-tasks", "1"])

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2

    assert "Ollama backend unavailable" in capsys.readouterr().err


def _response(answer: str) -> AgentResponse:
    return AgentResponse(
        reasoning="Brief reasoning.",
        final_answer=answer,
        metadata={"raw_output": '{"reasoning": "Brief reasoning.", "final_answer": "%s"}' % answer, "parse_status": "strict_json"},
    )
