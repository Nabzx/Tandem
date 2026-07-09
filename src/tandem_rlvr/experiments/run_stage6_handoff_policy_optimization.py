from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tandem_rlvr.agents.base import Agent, AgentResponse
from tandem_rlvr.agents.llm import (
    OllamaBackendError,
    OllamaGenerationConfig,
    OllamaJuniorAgent,
    OllamaSeniorAgent,
)
from tandem_rlvr.agents.llm.handoff_strategies import DEFAULT_HANDOFF_STRATEGY, get_handoff_strategy, list_handoff_strategy_names
from tandem_rlvr.experiments.run_stage5_generalization_eval import parse_splits
from tandem_rlvr.rl import compute_handoff_reward
from tandem_rlvr.rl.policy import create_bandit
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.splits import build_split_benchmark
from tandem_rlvr.tasks.verifiers import verify_final_answer
from tandem_rlvr.utils.io import ensure_parent_dir, write_json
from tandem_rlvr.utils.seed import seed_everything


STAGE6_NORMAL_NUM_PREDICT = 256
STAGE6_QUICK_NUM_PREDICT = 96
STAGE6_NORMAL_TIMEOUT_SECONDS = 120
STAGE6_QUICK_TIMEOUT_SECONDS = 60
STAGE6_DEFAULT_TEMPERATURE = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 RLVR-style handoff prompt-policy optimization.")
    parser.add_argument("--num-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--senior-model", type=str, default="llama3.1:latest")
    parser.add_argument("--junior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--splits", type=str, default="id_eval,ood_eval,stress_eval")
    parser.add_argument("--bandit", choices=["epsilon_greedy", "ucb1"], default="ucb1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--temperature", type=float, default=STAGE6_DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    num_predict, timeout_seconds, temperature = resolve_stage6_generation_settings(args)
    _print_start(args, num_predict, timeout_seconds, temperature)
    senior_config = OllamaGenerationConfig(
        model_name=args.senior_model,
        temperature=temperature,
        top_p=args.top_p,
        num_predict=num_predict,
        timeout_seconds=timeout_seconds,
    )
    junior_config = OllamaGenerationConfig(
        model_name=args.junior_model,
        temperature=temperature,
        top_p=args.top_p,
        num_predict=num_predict,
        timeout_seconds=timeout_seconds,
    )
    try:
        result = run_stage6_optimization(
            num_episodes=args.num_episodes,
            seed=args.seed,
            senior_model=args.senior_model,
            junior_model=args.junior_model,
            splits=parse_splits(args.splits),
            bandit_name=args.bandit,
            output_dir=args.output_dir,
            senior_agent=OllamaSeniorAgent(senior_config, check_availability=False),
            junior_agent=OllamaJuniorAgent(junior_config, check_availability=False),
            preflight=not args.skip_preflight,
            warmup=args.warmup,
            progress=True,
        )
    except OllamaBackendError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2) from None

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
    preflight: bool = False,
    warmup: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    if preflight:
        preflight_ollama_agents(senior_agent, junior_agent)
    warmup_seconds: dict[str, float] = {}
    if warmup:
        warmup_seconds = warmup_ollama_agents(senior_agent, junior_agent, progress=progress)

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
        if progress:
            print(
                f"Episode {episode}/{num_episodes} | split={task.metadata.get('split', '')} | "
                f"strategy={strategy_name} | task_type={task.task_type}",
                flush=True,
            )
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
        if progress:
            print(f"  senior generation: {_format_seconds(row['senior_generation_seconds'])}", flush=True)
            print(f"  junior generation: {_format_seconds(row['junior_generation_seconds'])}", flush=True)
            print(f"  correct={row['correct']} | reward={row['total_reward']}", flush=True)

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
            "warmup_seconds": warmup_seconds,
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
    senior_start = time.monotonic()
    try:
        senior_response = _produce_handoff(senior_agent, task, strategy_name)
    except Exception as exc:
        return _failed_episode_row(
            episode=episode,
            task=task,
            strategy_name=strategy_name,
            senior_model=senior_model,
            junior_model=junior_model,
            failure_type=_failure_type_from_exception(exc),
            error=str(exc),
            senior_generation_seconds=time.monotonic() - senior_start,
            junior_generation_seconds=None,
        )
    senior_generation_seconds = time.monotonic() - senior_start

    junior_start = time.monotonic()
    try:
        junior_response = junior_agent.answer(task, context=senior_response.reasoning)
    except Exception as exc:
        return _failed_episode_row(
            episode=episode,
            task=task,
            strategy_name=strategy_name,
            senior_model=senior_model,
            junior_model=junior_model,
            failure_type=_failure_type_from_exception(exc),
            error=str(exc),
            senior_response=senior_response,
            senior_generation_seconds=senior_generation_seconds,
            junior_generation_seconds=time.monotonic() - junior_start,
        )
    junior_generation_seconds = time.monotonic() - junior_start

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
        "senior_generation_seconds": senior_generation_seconds,
        "junior_generation_seconds": junior_generation_seconds,
        "parse_status": junior_response.metadata.get("parse_status", "unknown"),
        "failure_type": "" if correct else "wrong_answer",
        "error": "",
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


def preflight_ollama_agents(senior_agent: Agent, junior_agent: Agent) -> None:
    for agent in (senior_agent, junior_agent):
        check_available = getattr(agent, "check_available", None)
        if callable(check_available):
            check_available()
        check_model_available = getattr(agent, "check_model_available", None)
        if callable(check_model_available):
            check_model_available()


def warmup_ollama_agents(senior_agent: Agent, junior_agent: Agent, progress: bool = False) -> dict[str, float]:
    timings: dict[str, float] = {}
    for label, agent in (("senior", senior_agent), ("junior", junior_agent)):
        warmup_fn = getattr(agent, "warmup", None)
        if not callable(warmup_fn):
            continue
        if progress:
            model_name = getattr(agent, "model_name", label)
            print(f"Warming up {label} model: {model_name}", flush=True)
        timings[label] = float(warmup_fn())
        if progress:
            print(f"  {label} warmup: {_format_seconds(timings[label])}", flush=True)
    return timings


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


def _failed_episode_row(
    episode: int,
    task: Task,
    strategy_name: str,
    senior_model: str,
    junior_model: str,
    failure_type: str,
    error: str,
    senior_response: AgentResponse | None = None,
    senior_generation_seconds: float | None = None,
    junior_generation_seconds: float | None = None,
) -> dict[str, Any]:
    reward = -0.5 if failure_type == "timeout" else 0.0
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
        "junior_answer": "",
        "correct": False,
        "process_reward_score": 0.0,
        "legibility_score": 0.0,
        "leakage_score": 1.0,
        "relevance_score": 0.0,
        "usefulness_score": 0.0,
        "leaks_exact_answer": False,
        "hallucination_flag": False,
        "total_reward": reward,
        "reward_components": json.dumps({"failure_type": failure_type, "error": error}, sort_keys=True),
        "senior_handoff_reasoning": "" if senior_response is None else senior_response.reasoning,
        "raw_senior_output": "" if senior_response is None else senior_response.metadata.get("raw_output", ""),
        "raw_junior_output": "",
        "senior_generation_seconds": senior_generation_seconds,
        "junior_generation_seconds": junior_generation_seconds,
        "parse_status": "backend_error",
        "failure_type": failure_type,
        "error": error,
    }


def _failure_type_from_exception(exc: Exception) -> str:
    failure_type = getattr(exc, "failure_type", None)
    if failure_type in {"backend_unavailable", "model_not_found", "timeout", "backend_error"}:
        return str(failure_type)
    message = str(exc).lower()
    if "timeout" in message or "timed out" in message:
        return "timeout"
    return "backend_error"


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


def resolve_stage6_generation_settings(args: argparse.Namespace) -> tuple[int, int, float]:
    if args.num_predict is not None:
        num_predict = args.num_predict
    elif args.quick:
        num_predict = STAGE6_QUICK_NUM_PREDICT
    else:
        num_predict = STAGE6_NORMAL_NUM_PREDICT

    if args.max_generation_seconds is not None:
        timeout_seconds = args.max_generation_seconds
    elif getattr(args, "timeout_seconds", None) is not None:
        timeout_seconds = args.timeout_seconds
    elif args.quick:
        timeout_seconds = STAGE6_QUICK_TIMEOUT_SECONDS
    else:
        timeout_seconds = STAGE6_NORMAL_TIMEOUT_SECONDS

    return num_predict, timeout_seconds, float(args.temperature)


def _print_start(args: argparse.Namespace, num_predict: int, timeout_seconds: int, temperature: float) -> None:
    print("Starting Stage 6 handoff policy optimization", flush=True)
    print(f"Senior model: {args.senior_model}", flush=True)
    print(f"Junior model: {args.junior_model}", flush=True)
    print(f"Number of episodes: {args.num_episodes}", flush=True)
    print(f"Splits: {args.splits}", flush=True)
    print(f"Bandit: {args.bandit}", flush=True)
    print(f"num_predict: {num_predict}", flush=True)
    print(f"max generation seconds: {timeout_seconds}", flush=True)
    print(f"temperature: {temperature}", flush=True)


def _format_seconds(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.1f}s"


if __name__ == "__main__":
    main()
