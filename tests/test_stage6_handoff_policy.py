from pathlib import Path

import pytest

from tandem_rlvr.agents.base import AgentResponse
from tandem_rlvr.agents.llm.handoff_strategies import HANDOFF_STRATEGIES, get_handoff_strategy, list_handoff_strategy_names
from tandem_rlvr.agents.llm.prompts import senior_handoff_prompt
from tandem_rlvr.experiments.run_stage6_handoff_policy_optimization import (
    run_stage6_optimization,
    run_strategy_episode,
    summarize_strategy_eval,
)
from tandem_rlvr.rl import EpsilonGreedyBandit, UCB1Bandit, compute_handoff_reward
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.splits import build_split_benchmark


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
    )

    assert result["episodes_path"].exists()
    assert result["summary_path"].exists()
    assert result["strategy_eval_path"].exists()
    assert result["strategy_eval_summary_path"].exists()
    assert result["summary"]["best_strategy"] in list_handoff_strategy_names()
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
    )

    summary = summarize_strategy_eval(result["strategy_eval"])

    assert "accuracy_by_strategy" in summary
    assert "process_reward_by_strategy" in summary
    assert "accuracy_by_split_and_strategy" in summary
    assert "id_eval" in summary["accuracy_by_split_and_strategy"]
