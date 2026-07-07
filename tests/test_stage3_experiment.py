from pathlib import Path

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.agents.llm import OllamaBackendUnavailable
from tandem_rlvr.experiments.run_stage3_llm_eval import main, run_stage3_llm_eval
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
    )

    assert len(result["results"]) == 8
    assert result["summary"]["total_tasks"] == 2
    assert result["summary"]["senior_model"] == "fake-senior"
    assert (tmp_path / "stage3_llm_results.csv").exists()
    assert (tmp_path / "stage3_llm_summary.json").exists()


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
