from pathlib import Path
from argparse import Namespace

import pytest

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.agents.llm import OllamaGenerationTimeout, OllamaModelNotFound
from tandem_rlvr.agents.llm.handoff_strategies import HANDOFF_STRATEGIES, get_handoff_strategy, list_handoff_strategy_names
from tandem_rlvr.agents.llm.prompts import senior_handoff_prompt
from tandem_rlvr.experiments.run_stage6_handoff_policy_optimization import (
    build_episode_summary_table,
    build_heldout_summary_table,
    preflight_ollama_agents,
    resolve_stage6_generation_settings,
    run_stage6_optimization,
    run_strategy_episode,
    summarize_bandit,
    summarize_strategy_eval,
    warmup_ollama_agents,
)
from tandem_rlvr.rl import EpsilonGreedyBandit, UCB1Bandit, compute_handoff_reward
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.splits import build_split_benchmark
import pandas as pd


class FakeSenior:
    def produce_handoff(self, task, strategy_name="structured_steps"):
        strategy = get_handoff_strategy(strategy_name)
        return AgentResponse(
            reasoning=f"{strategy.name}: {strategy.instruction} Use the operation from the prompt.",
            final_answer="",
            metadata={
                "raw_output": '{"reasoning": "handoff", "final_answer": null}',
                "parse_status": "strict_json",
            },
        )


class FakeJunior:
    def answer(self, task, context=None):
        return AgentResponse(
            reasoning="Use the supplied hint and solve directly.",
            final_answer=str(task.answer),
            metadata={
                "raw_output": '{"reasoning": "done", "final_answer": "%s"}' % task.answer,
                "parse_status": "strict_json",
            },
        )


class TimeoutOnceJunior(FakeJunior):
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, task, context=None):
        self.calls += 1
        if self.calls == 1:
            raise OllamaGenerationTimeout(
                "Ollama generation timed out after 60 seconds. Try increasing --max-generation-seconds."
            )
        return super().answer(task, context)


class FakeWarmupAgent:
    def __init__(self) -> None:
        self.warmup_calls = 0

    def warmup(self) -> float:
        self.warmup_calls += 1
        return 0.25


class MissingModelAgent:
    def check_available(self) -> None:
        return None

    def check_model_available(self) -> None:
        raise OllamaModelNotFound("Model not found. Run: ollama pull missing-model")


def test_handoff_strategy_registry_contains_stage6_strategies() -> None:
    expected = {
        "minimal_hint",
        "structured_steps",
        "worked_prefix",
        "verification_hint",
        "anti_hallucination",
        "direct_teaching",
    }

    assert set(list_handoff_strategy_names()) == expected
    assert set(HANDOFF_STRATEGIES) == expected
    assert get_handoff_strategy("verification_hint").instruction


def test_senior_handoff_prompt_injects_strategy_instruction() -> None:
    task = Task(
        task_id="t1",
        task_type="multiplication",
        prompt="What is 7 * 8?",
        answer="56",
        difficulty="easy",
        metadata={"task_family": "arithmetic"},
    )
    strategy = get_handoff_strategy("minimal_hint")

    prompt = senior_handoff_prompt(task, strategy.name, strategy.instruction)

    assert "Handoff strategy: minimal_hint." in prompt
    assert f"Strategy instruction: {strategy.instruction}" in prompt
    assert '"final_answer": null' in prompt
    assert "Do not reveal the final answer." in prompt


def test_handoff_reward_components_and_penalties() -> None:
    base = compute_handoff_reward(
        tandem_correct=True,
        process_metrics={
            "process_reward_score": 0.8,
            "usefulness_score": 0.9,
            "leaks_exact_answer": False,
            "hallucination_flags": [],
        },
    )
    leaked = compute_handoff_reward(
        tandem_correct=True,
        process_metrics={
            "process_reward_score": 0.8,
            "usefulness_score": 0.9,
            "leaks_exact_answer": True,
            "hallucination_flags": [],
        },
    )
    hallucinated = compute_handoff_reward(
        tandem_correct=True,
        process_metrics={
            "process_reward_score": 0.8,
            "usefulness_score": 0.9,
            "leaks_exact_answer": False,
            "hallucination_flags": ["external data"],
        },
    )

    assert base.total_reward == pytest.approx(1.42)
    assert leaked.total_reward < base.total_reward
    assert hallucinated.total_reward < base.total_reward
    assert leaked.leakage_penalty == 0.5
    assert hallucinated.hallucination_penalty == 0.3


def test_bandits_select_and_update_actions() -> None:
    epsilon = EpsilonGreedyBandit(actions=["a", "b"], epsilon=0.0, seed=0)
    assert epsilon.select_action() == "a"
    epsilon.update("b", 2.0)
    assert epsilon.select_action() == "b"

    ucb = UCB1Bandit(actions=["a", "b"])
    assert ucb.select_action() == "a"
    ucb.update("a", 1.0)
    assert ucb.select_action() == "b"


def test_stage6_episode_logging_with_mocked_agents() -> None:
    task = build_split_benchmark(1, splits=["id_eval"], seed=42, task_families=["arithmetic"])[0]

    row = run_strategy_episode(
        episode=1,
        task=task,
        strategy_name="structured_steps",
        senior_model="fake-senior",
        junior_model="fake-junior",
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
    )

    assert row["episode"] == 1
    assert row["strategy"] == "structured_steps"
    assert row["split"] == "id_eval"
    assert row["correct"] is True
    assert row["total_reward"] > 0
    assert row["senior_handoff_reasoning"]
    assert row["raw_senior_output"]
    assert row["raw_junior_output"]
    assert row["senior_generation_seconds"] is not None
    assert row["junior_generation_seconds"] is not None


def test_stage6_timeout_becomes_failure_row() -> None:
    task = build_split_benchmark(1, splits=["id_eval"], seed=42, task_families=["arithmetic"])[0]

    row = run_strategy_episode(
        episode=1,
        task=task,
        strategy_name="structured_steps",
        senior_model="fake-senior",
        junior_model="fake-junior",
        senior_agent=FakeSenior(),
        junior_agent=TimeoutOnceJunior(),
    )

    assert row["correct"] is False
    assert row["failure_type"] == "timeout"
    assert row["total_reward"] == -0.5
    assert "timed out" in row["error"]


def test_stage6_one_timeout_does_not_crash_full_run(tmp_path: Path) -> None:
    result = run_stage6_optimization(
        num_episodes=2,
        seed=7,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval"],
        bandit_name="ucb1",
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=TimeoutOnceJunior(),
    )

    episodes = result["episodes"]
    assert len(episodes) == 2
    assert "timeout" in set(episodes["failure_type"])
    assert result["episodes_path"].exists()


def test_stage6_optimization_writes_outputs_and_summary(tmp_path: Path) -> None:
    result = run_stage6_optimization(
        num_episodes=3,
        seed=7,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval"],
        bandit_name="epsilon_greedy",
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        heldout_tasks_per_split=1,
        eval_strategies="best,default",
    )

    assert result["episodes_path"].exists()
    assert result["summary_path"].exists()
    assert result["strategy_eval_path"].exists()
    assert result["strategy_eval_summary_path"].exists()
    assert result["summary"]["best_strategy"] in list_handoff_strategy_names()
    assert result["summary"]["episode_best_strategy"] in list_handoff_strategy_names()
    assert result["summary"]["heldout_best_strategy"] in list_handoff_strategy_names()
    assert result["summary"]["bandit"] == "epsilon_greedy"
    assert result["summary"]["heldout_splits"] == ["id_eval"]
    assert "heldout_accuracy_by_split" in result["summary"]
    assert result["summary"]["heldout_accuracy_by_split"]["id_eval"] == 1.0
    assert "default_strategy_comparison" in result["summary"]


def test_stage6_strategy_eval_summary_contains_heldout_views(tmp_path: Path) -> None:
    result = run_stage6_optimization(
        num_episodes=2,
        seed=11,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval"],
        bandit_name="ucb1",
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        heldout_tasks_per_split=1,
    )

    summary = summarize_strategy_eval(result["strategy_eval"])

    assert "accuracy_by_strategy" in summary
    assert "process_reward_by_strategy" in summary
    assert "accuracy_by_split_and_strategy" in summary
    assert "id_eval" in summary["accuracy_by_split_and_strategy"]


def test_stage6_generation_settings_use_stage6_quick_defaults_and_overrides() -> None:
    quick_args = Namespace(num_predict=None, quick=True, timeout_seconds=None, max_generation_seconds=None, temperature=0.0)
    normal_args = Namespace(num_predict=None, quick=False, timeout_seconds=None, max_generation_seconds=None, temperature=0.0)
    override_args = Namespace(num_predict=192, quick=True, timeout_seconds=None, max_generation_seconds=120, temperature=0.0)

    assert resolve_stage6_generation_settings(quick_args) == (96, 60, 0.0)
    assert resolve_stage6_generation_settings(normal_args) == (256, 120, 0.0)
    assert resolve_stage6_generation_settings(override_args) == (192, 120, 0.0)


def test_stage6_warmup_can_be_mocked() -> None:
    senior = FakeWarmupAgent()
    junior = FakeWarmupAgent()

    timings = warmup_ollama_agents(senior, junior)

    assert timings == {"senior": 0.25, "junior": 0.25}
    assert senior.warmup_calls == 1
    assert junior.warmup_calls == 1


def test_stage6_preflight_model_missing_message() -> None:
    with pytest.raises(OllamaModelNotFound, match="Model not found. Run: ollama pull missing-model"):
        preflight_ollama_agents(MissingModelAgent(), FakeSenior())


def test_stage6_summary_distinguishes_episode_best_from_heldout_best() -> None:
    episodes = pd.DataFrame(
        [
            _summary_row("worked_prefix", True, 1.4, 0.6),
            _summary_row("worked_prefix", True, 1.2, 0.6),
            _summary_row("verification_hint", False, 0.3, 0.8),
        ]
    )
    heldout = pd.DataFrame(
        [
            _summary_row("worked_prefix", True, 1.0, 0.5),
            _summary_row("verification_hint", True, 1.6, 0.9),
            _summary_row("structured_steps", True, 1.1, 0.6),
        ]
    )

    summary = summarize_bandit(episodes, heldout, "worked_prefix")

    assert summary["episode_best_strategy"] == "worked_prefix"
    assert summary["episode_best_strategy_mean_reward"] == pytest.approx(1.3)
    assert summary["heldout_best_strategy"] == "verification_hint"
    assert summary["heldout_best_strategy_mean_reward"] == pytest.approx(1.6)
    assert summary["heldout_best_strategy_accuracy"] == 1.0
    assert summary["best_strategy"] == "worked_prefix"


def test_stage6_warns_when_episodes_less_than_strategy_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = run_stage6_optimization(
        num_episodes=5,
        seed=7,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval"],
        bandit_name="ucb1",
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        progress=True,
        heldout_tasks_per_split=1,
        eval_strategies="best,default",
    )

    output = capsys.readouterr().out
    assert "num_episodes is smaller than the number of strategies" in output
    assert "very small optimization run" in output
    assert any("smaller than the number of strategies" in warning for warning in result["summary"]["warnings"])


def test_stage6_warns_when_run_is_very_small(tmp_path: Path) -> None:
    result = run_stage6_optimization(
        num_episodes=12,
        seed=7,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval"],
        bandit_name="ucb1",
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        heldout_tasks_per_split=1,
        eval_strategies="best,default",
    )

    assert result["summary"]["warnings"] == ["Warning: this is a very small optimization run. Strategy rankings may be noisy."]


def test_stage6_force_initial_exploration_selects_unique_strategies_first(tmp_path: Path) -> None:
    strategies = list_handoff_strategy_names()

    result = run_stage6_optimization(
        num_episodes=len(strategies),
        seed=7,
        senior_model="fake-senior",
        junior_model="fake-junior",
        splits=["id_eval"],
        bandit_name="epsilon_greedy",
        output_dir=tmp_path,
        senior_agent=FakeSenior(),
        junior_agent=FakeJunior(),
        heldout_tasks_per_split=1,
        eval_strategies="best,default",
    )

    assert list(result["episodes"]["strategy"]) == strategies


def test_stage6_default_comparison_has_deltas() -> None:
    episodes = pd.DataFrame([_summary_row("verification_hint", True, 1.0, 0.8)])
    heldout = pd.DataFrame(
        [
            _summary_row("verification_hint", True, 1.5, 0.9),
            _summary_row("structured_steps", False, 0.5, 0.4),
        ]
    )

    summary = summarize_bandit(episodes, heldout, "verification_hint")
    comparison = summary["default_strategy_comparison"]

    assert comparison["heldout_best_vs_default_accuracy_delta"] == 1.0
    assert comparison["heldout_best_vs_default_reward_delta"] == 1.0
    assert comparison["heldout_best_vs_default_process_reward_delta"] == pytest.approx(0.5)


def test_stage6_printed_summary_tables_have_expected_columns() -> None:
    episodes = pd.DataFrame(
        [
            _summary_row("minimal_hint", True, 1.0, 0.7),
            _summary_row("minimal_hint", False, 0.0, 0.2),
        ]
    )
    heldout = pd.DataFrame([_summary_row("structured_steps", True, 1.2, 0.8)])

    episode_table = build_episode_summary_table(episodes)
    heldout_table = build_heldout_summary_table(heldout)

    assert list(episode_table.columns) == [
        "strategy",
        "selected_count",
        "episode_accuracy",
        "episode_mean_reward",
        "episode_process_reward",
        "episode_leakage_rate",
        "episode_hallucination_rate",
    ]
    assert list(heldout_table.columns) == [
        "strategy",
        "heldout_accuracy",
        "heldout_mean_reward",
        "heldout_process_reward",
        "heldout_leakage_rate",
        "heldout_hallucination_rate",
    ]


def _summary_row(strategy: str, correct: bool, total_reward: float, process_reward: float) -> dict[str, object]:
    return {
        "strategy": strategy,
        "correct": correct,
        "total_reward": total_reward,
        "process_reward_score": process_reward,
        "leaks_exact_answer": False,
        "hallucination_flag": False,
        "split": "id_eval",
    }
