from pathlib import Path

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.experiments.run_stage5_generalization_eval import run_stage5_generalization_eval


class FakeSenior:
    def answer(self, task, context=None):
        return _response(task.answer)

    def produce_handoff(self, task):
        return AgentResponse(
            reasoning=f"Use the task operation carefully. The answer is {task.answer}.",
            final_answer="",
            metadata={"raw_output": '{"reasoning": "handoff", "final_answer": null}', "parse_status": "strict_json"},
        )


class FakeJunior:
    def answer(self, task, context=None):
        if context and "[NUM]" not in context:
            return _response(task.answer)
        if task.metadata["split"] == "id_eval":
            return _response(task.answer)
        return _response("wrong")


def test_stage5_script_smoke_with_fake_agents(tmp_path: Path) -> None:
    result = run_stage5_generalization_eval(
        num_tasks_per_split=2,
        seed=42,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval", "ood_eval"],
        task_families=["arithmetic", "list"],
        modes=["senior_only", "junior_only", "tandem_handoff", "corrupted_handoff"],
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
    )

    assert result["results_path"].exists()
    assert result["summary_path"].exists()
    assert result["process_metrics_path"].exists()
    assert result["process_summary_path"].exists()
    assert set(result["results"]["split"]) == {"id_eval", "ood_eval"}
    assert "accuracy_by_split_and_mode" in result["summary"]
    assert "process_reward_by_split" in result["summary"]


def test_stage5_summary_nulls_and_generalization_gaps(tmp_path: Path) -> None:
    result = run_stage5_generalization_eval(
        num_tasks_per_split=1,
        seed=0,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval", "ood_eval"],
        task_families=["arithmetic"],
        modes=["tandem_handoff"],
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
    )

    summary = result["summary"]
    assert summary["accuracy_by_split_and_mode"]["id_eval"]["junior_only"] is None
    assert summary["handoff_gain_by_split"]["id_eval"] is None
    assert summary["robustness_drop_by_split"]["id_eval"] is None
    assert summary["ood_generalization_gap"] is not None
    assert summary["stress_generalization_gap"] is None


def _response(answer: str) -> AgentResponse:
    return AgentResponse(
        reasoning="Brief reasoning.",
        final_answer=answer,
        metadata={"raw_output": '{"reasoning": "Brief reasoning.", "final_answer": "%s"}' % answer, "parse_status": "strict_json"},
    )
