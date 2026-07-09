from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from tandem_rlvr.agents.base import Agent, AgentResponse
from tandem_rlvr.agents.llm import OllamaGenerationConfig, OllamaJuniorAgent, OllamaSeniorAgent
from tandem_rlvr.agents.llm.handoff_strategies import DEFAULT_HANDOFF_STRATEGY, get_handoff_strategy, list_handoff_strategy_names
from tandem_rlvr.experiments.run_stage3_llm_eval import resolve_generation_settings
from tandem_rlvr.experiments.run_stage5_generalization_eval import parse_splits
from tandem_rlvr.rl import compute_handoff_reward
from tandem_rlvr.rl.policy import create_bandit
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.splits import build_split_benchmark
from tandem_rlvr.tasks.verifiers import verify_final_answer
from tandem_rlvr.utils.io import ensure_parent_dir, write_json
from tandem_rlvr.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 RLVR-style handoff prompt-policy optimization.")
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--senior-model", type=str, default="llama3.1:latest")
    parser.add_argument("--junior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--splits", type=str, default="id_eval,ood_eval,stress_eval")
    parser.add_argument("--bandit", choices=["epsilon_greedy", "ucb1"], default="ucb1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    num_predict, timeout_seconds = resolve_generation_settings(args)
    senior_config = OllamaGenerationConfig(args.senior_model, args.temperature, args.top_p, num_predict, timeout_seconds)
    junior_config = OllamaGenerationConfig(args.junior_model, args.temperature, args.top_p, num_predict, timeout_seconds)
    result = run_stage6_optimization(
        num_episodes=args.num_episodes,
        seed=args.seed,
        senior_model=args.senior_model,
        junior_model=args.junior_model,
        splits=parse_splits(args.splits),
        bandit_name=args.bandit,
        output_dir=args.output_dir,
        senior_agent=OllamaSeniorAgent(senior_config),
        junior_agent=OllamaJuniorAgent(junior_config),
    )

    print("\nStage 6 handoff policy optimization summary")
    print(pd.DataFrame([_flat_summary(result["summary"])]).to_string(index=False))
    print(f"\nWrote episode logs to {result['episodes_path']}")
    print(f"Wrote bandit summary to {result['summary_path']}")
    print(f"Wrote strategy eval to {result['strategy_eval_path']}")
    print(f"Wrote strategy eval summary to {result['strategy_eval_summary_path']}")


def run_stage6_optimization(
    num_episodes: int,
    seed: int,
    senior_model: str,
    junior_model: str,
    splits: list[str],
    bandit_name: str,
    output_dir: str | Path,
    senior_agent: Agent,
    junior_agent: Agent,
) -> dict[str, Any]:
    rng = random.Random(seed)
    strategies = list_handoff_strategy_names()
    bandit = create_bandit(bandit_name, strategies, seed=seed)
    tasks = build_split_benchmark(max(1, num_episodes // max(1, len(splits))), splits=splits, seed=seed)
    if len(tasks) < num_episodes:
        tasks.extend(build_split_benchmark(num_episodes - len(tasks), splits=splits, seed=seed + 999))

    episode_rows: list[dict[str, Any]] = []
    for episode in range(1, num_episodes + 1):
        task = rng.choice(tasks)
        strategy_name = bandit.select_action(context=_context(task))
        row = run_strategy_episode(
            episode=episode,
            task=task,
            strategy_name=strategy_name,
            senior_model=senior_model,
            junior_model=junior_model,
            senior_agent=senior_agent,
            junior_agent=junior_agent,
        )
        bandit.update(strategy_name, float(row["total_reward"]), context=_context(task))
        episode_rows.append(row)

    episodes = pd.DataFrame(episode_rows)
    best_strategy = _best_strategy(episodes)
    heldout_splits = [split for split in splits if split != "train"] or splits
    heldout_tasks = build_split_benchmark(5, splits=heldout_splits, seed=seed + 2024)
    strategy_eval = evaluate_strategies(
        tasks=heldout_tasks,
        strategies=[best_strategy, DEFAULT_HANDOFF_STRATEGY, *[s for s in strategies if s not in {best_strategy, DEFAULT_HANDOFF_STRATEGY}]],
        senior_model=senior_model,
        junior_model=junior_model,
        senior_agent=senior_agent,
        junior_agent=junior_agent,
    )

    summary = summarize_bandit(episodes, strategy_eval, best_strategy)
    summary.update(
        {
            "num_episodes": num_episodes,
            "senior_model": senior_model,
            "junior_model": junior_model,
            "splits": splits,
            "heldout_splits": heldout_splits,
            "bandit": bandit_name,
            "strategies": strategies,
        }
    )
    strategy_summary = summarize_strategy_eval(strategy_eval)
    output_dir = Path(output_dir)
    episodes_path = output_dir / "stage6_bandit_episodes.csv"
    summary_path = output_dir / "stage6_bandit_summary.json"
    strategy_eval_path = output_dir / "stage6_strategy_eval.csv"
    strategy_eval_summary_path = output_dir / "stage6_strategy_eval_summary.json"
    ensure_parent_dir(episodes_path)
    episodes.to_csv(episodes_path, index=False)
    strategy_eval.to_csv(strategy_eval_path, index=False)
    write_json(summary, summary_path)
    write_json(strategy_summary, strategy_eval_summary_path)
    return {
        "episodes": episodes,
        "summary": summary,
        "strategy_eval": strategy_eval,
        "strategy_eval_summary": strategy_summary,
        "episodes_path": episodes_path,
        "summary_path": summary_path,
        "strategy_eval_path": strategy_eval_path,
        "strategy_eval_summary_path": strategy_eval_summary_path,
    }


def run_strategy_episode(
    episode: int,
    task: Task,
    strategy_name: str,
    senior_model: str,
    junior_model: str,
    senior_agent: Agent,
    junior_agent: Agent,
) -> dict[str, Any]:
    senior_response = _produce_handoff(senior_agent, task, strategy_name)
    junior_response = junior_agent.answer(task, context=senior_response.reasoning)
    correct = verify_final_answer(task, junior_response.final_answer)
    metrics = _process_metrics_for_episode(task, senior_response.reasoning, correct)
    reward = compute_handoff_reward(correct, metrics)
    hallucination_flag = bool(metrics.get("hallucination_flags"))
    return {
        "episode": episode,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "task_family": task.metadata.get("task_family", ""),
        "split": task.metadata.get("split", ""),
        "difficulty": task.difficulty,
        "strategy": strategy_name,
        "senior_model": senior_model,
        "junior_model": junior_model,
        "expected_answer": task.answer,
        "junior_answer": junior_response.final_answer,
        "correct": correct,
        "process_reward_score": metrics["process_reward_score"],
        "legibility_score": metrics["legibility_score"],
        "leakage_score": metrics["leakage_score"],
        "relevance_score": metrics["relevance_score"],
        "usefulness_score": metrics["usefulness_score"],
        "leaks_exact_answer": metrics["leaks_exact_answer"],
        "hallucination_flag": hallucination_flag,
        "total_reward": reward.total_reward,
        "reward_components": json.dumps(reward.components, sort_keys=True),
        "senior_handoff_reasoning": senior_response.reasoning,
        "raw_senior_output": senior_response.metadata.get("raw_output", ""),
        "raw_junior_output": junior_response.metadata.get("raw_output", ""),
        "parse_status": junior_response.metadata.get("parse_status", "unknown"),
        "failure_type": "" if correct else "wrong_answer",
    }


def evaluate_strategies(
    tasks: list[Task],
    strategies: list[str],
    senior_model: str,
    junior_model: str,
    senior_agent: Agent,
    junior_agent: Agent,
) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for strategy_name in strategies:
        if strategy_name in seen:
            continue
        seen.add(strategy_name)
        for index, task in enumerate(tasks, start=1):
            row = run_strategy_episode(index, task, strategy_name, senior_model, junior_model, senior_agent, junior_agent)
            row.pop("episode")
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_bandit(episodes: pd.DataFrame, strategy_eval: pd.DataFrame, best_strategy: str) -> dict[str, Any]:
    return {
        "mean_reward": _mean(episodes, "total_reward"),
        "mean_reward_by_strategy": _mean_by(episodes, "strategy", "total_reward"),
        "selection_counts_by_strategy": _counts(episodes, "strategy"),
        "accuracy_by_strategy": _accuracy_by(episodes, "strategy"),
        "process_reward_by_strategy": _mean_by(episodes, "strategy", "process_reward_score"),
        "leakage_rate_by_strategy": _rate_by(episodes, "strategy", "leaks_exact_answer"),
        "hallucination_rate_by_strategy": _rate_by(episodes, "strategy", "hallucination_flag"),
        "best_strategy": best_strategy,
        "best_strategy_mean_reward": _mean(strategy_eval[strategy_eval["strategy"] == best_strategy], "total_reward"),
        "heldout_accuracy_by_split": _accuracy_by(strategy_eval[strategy_eval["strategy"] == best_strategy], "split"),
        "heldout_process_reward_by_split": _mean_by(strategy_eval[strategy_eval["strategy"] == best_strategy], "split", "process_reward_score"),
        "default_strategy_comparison": _default_comparison(strategy_eval, best_strategy),
    }


def summarize_strategy_eval(strategy_eval: pd.DataFrame) -> dict[str, Any]:
    return {
        "mean_reward_by_strategy": _mean_by(strategy_eval, "strategy", "total_reward"),
        "accuracy_by_strategy": _accuracy_by(strategy_eval, "strategy"),
        "process_reward_by_strategy": _mean_by(strategy_eval, "strategy", "process_reward_score"),
        "accuracy_by_split_and_strategy": {
            split: _accuracy_by(group, "strategy")
            for split, group in strategy_eval.groupby("split")
        },
    }


def _process_metrics_for_episode(task: Task, reasoning: str, correct: bool) -> dict[str, Any]:
    from tandem_rlvr.metrics import compute_legibility_metrics, compute_leakage_metrics, compute_process_reward, compute_relevance_metrics

    legibility = compute_legibility_metrics(task, reasoning)
    leakage = compute_leakage_metrics(task, reasoning)
    relevance = compute_relevance_metrics(task, reasoning)
    usefulness_score = 0.75 if correct else 0.35
    components = {**legibility, **leakage, **relevance, "usefulness_score": usefulness_score}
    reward = compute_process_reward(components)
    return {**components, **reward}


def _produce_handoff(senior_agent: Agent, task: Task, strategy_name: str) -> AgentResponse:
    handoff_fn = getattr(senior_agent, "produce_handoff", None)
    if callable(handoff_fn):
        try:
            return handoff_fn(task, strategy_name=strategy_name)
        except TypeError:
            return handoff_fn(task)
    return senior_agent.answer(task)


def _context(task: Task) -> dict[str, str]:
    return {
        "task_type": task.task_type,
        "task_family": str(task.metadata.get("task_family", "")),
        "split": str(task.metadata.get("split", "")),
    }


def _best_strategy(episodes: pd.DataFrame) -> str:
    means = episodes.groupby("strategy")["total_reward"].mean()
    return str(means.idxmax())


def _mean(df: pd.DataFrame, column: str) -> float | None:
    if df.empty:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def _mean_by(df: pd.DataFrame, group_column: str, value_column: str) -> dict[str, float]:
    return {
        str(group): float(pd.to_numeric(group_df[value_column], errors="coerce").mean())
        for group, group_df in df.groupby(group_column)
    }


def _accuracy_by(df: pd.DataFrame, group_column: str) -> dict[str, float]:
    return {
        str(group): float(group_df["correct"].mean())
        for group, group_df in df.groupby(group_column)
    }


def _rate_by(df: pd.DataFrame, group_column: str, value_column: str) -> dict[str, float]:
    return {
        str(group): float(group_df[value_column].fillna(False).astype(bool).mean())
        for group, group_df in df.groupby(group_column)
    }


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in df[column].value_counts().sort_index().items()}


def _default_comparison(strategy_eval: pd.DataFrame, best_strategy: str) -> dict[str, Any]:
    default_rows = strategy_eval[strategy_eval["strategy"] == DEFAULT_HANDOFF_STRATEGY]
    best_rows = strategy_eval[strategy_eval["strategy"] == best_strategy]
    return {
        "default_strategy": DEFAULT_HANDOFF_STRATEGY,
        "best_strategy": best_strategy,
        "default_mean_reward": _mean(default_rows, "total_reward"),
        "best_mean_reward": _mean(best_rows, "total_reward"),
        "default_accuracy": None if default_rows.empty else float(default_rows["correct"].mean()),
        "best_accuracy": None if best_rows.empty else float(best_rows["correct"].mean()),
    }


def _flat_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_reward": summary["mean_reward"],
        "best_strategy": summary["best_strategy"],
        "best_strategy_mean_reward": summary["best_strategy_mean_reward"],
    }


if __name__ == "__main__":
    main()
