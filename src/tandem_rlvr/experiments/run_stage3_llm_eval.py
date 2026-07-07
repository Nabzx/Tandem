from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from tandem_rlvr.agents.base import Agent, AgentResponse
from tandem_rlvr.agents.llm import (
    OllamaBackendUnavailable,
    OllamaGenerationConfig,
    OllamaJuniorAgent,
    OllamaSeniorAgent,
)
from tandem_rlvr.eval.perturbations import light_noise
from tandem_rlvr.experiments.run_stage2_baseline import build_mixed_benchmark
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.tasks.verifiers import verify_final_answer
from tandem_rlvr.utils.io import ensure_parent_dir, write_json
from tandem_rlvr.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 TandemRLVR evaluation with local Ollama models.")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--senior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--junior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--quick", action="store_true", help="Use shorter generations for faster local smoke tests.")
    args = parser.parse_args()

    try:
        result = run_llm_eval_from_args(args)
    except OllamaBackendUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None

    print("\nStage 3 local LLM evaluation summary")
    print(pd.DataFrame([_flat_summary(result["summary"])]).to_string(index=False))
    print(f"\nWrote per-example results to {result['results_path']}")
    print(f"Wrote summary metrics to {result['summary_path']}")


def run_llm_eval_from_args(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    num_predict = 96 if args.quick else args.num_predict
    timeout_seconds = min(args.timeout_seconds, 60) if args.quick else args.timeout_seconds
    senior_config = OllamaGenerationConfig(
        model_name=args.senior_model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=num_predict,
        timeout_seconds=timeout_seconds,
    )
    junior_config = OllamaGenerationConfig(
        model_name=args.junior_model,
        temperature=args.temperature,
        top_p=args.top_p,
        num_predict=num_predict,
        timeout_seconds=timeout_seconds,
    )
    senior_agent, junior_agent = create_ollama_agents(senior_config, junior_config)
    tasks = build_mixed_benchmark(args.num_tasks, seed=args.seed, difficulty=args.difficulty)
    output_dir = Path(args.output_dir)
    results_path = output_dir / "stage3_llm_results.csv"
    summary_path = output_dir / "stage3_llm_summary.json"
    return run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=senior_agent,
        junior_agent=junior_agent,
        senior_model=args.senior_model,
        junior_model=args.junior_model,
        seed=args.seed,
        results_path=results_path,
        summary_path=summary_path,
    )


def create_ollama_agents(
    senior_config: OllamaGenerationConfig,
    junior_config: OllamaGenerationConfig,
) -> tuple[OllamaSeniorAgent, OllamaJuniorAgent]:
    return OllamaSeniorAgent(senior_config), OllamaJuniorAgent(junior_config)


def run_stage3_llm_eval(
    tasks: list[Task],
    senior_agent: Agent,
    junior_agent: Agent,
    senior_model: str,
    junior_model: str,
    seed: int,
    results_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        senior_only = senior_agent.answer(task)
        rows.append(_row(task, "senior_only", senior_model, junior_model, senior_only, senior_reasoning=senior_only.reasoning))

        junior_only = junior_agent.answer(task)
        rows.append(_row(task, "junior_only", senior_model, junior_model, junior_only))

        senior_handoff = _produce_handoff(senior_agent, task)
        tandem = junior_agent.answer(task, context=senior_handoff.reasoning)
        rows.append(
            _row(
                task,
                "tandem_handoff",
                senior_model,
                junior_model,
                tandem,
                senior_reasoning=senior_handoff.reasoning,
            )
        )

        corrupted_reasoning = light_noise(senior_handoff.reasoning, rng=rng)
        corrupted = junior_agent.answer(task, context=corrupted_reasoning)
        rows.append(
            _row(
                task,
                "corrupted_handoff",
                senior_model,
                junior_model,
                corrupted,
                senior_reasoning=senior_handoff.reasoning,
                corrupted_reasoning=corrupted_reasoning,
            )
        )

    results = pd.DataFrame(rows)
    summary = summarize_stage3_results(results, senior_model=senior_model, junior_model=junior_model)
    results_path = Path(results_path)
    ensure_parent_dir(results_path)
    results.to_csv(results_path, index=False)
    write_json(summary, summary_path)
    return {"results": results, "summary": summary, "results_path": results_path, "summary_path": Path(summary_path)}


def summarize_stage3_results(results: pd.DataFrame, senior_model: str, junior_model: str) -> dict[str, Any]:
    mode_acc = {
        mode: _accuracy_for(results[results["mode"] == mode])
        for mode in ["senior_only", "junior_only", "tandem_handoff", "corrupted_handoff"]
    }
    task_level = results[results["mode"] == "tandem_handoff"]
    return {
        "total_tasks": int(len(task_level)),
        "senior_model": senior_model,
        "junior_model": junior_model,
        "senior_only_accuracy": mode_acc["senior_only"],
        "junior_only_accuracy": mode_acc["junior_only"],
        "tandem_handoff_accuracy": mode_acc["tandem_handoff"],
        "corrupted_handoff_accuracy": mode_acc["corrupted_handoff"],
        "handoff_gain": mode_acc["tandem_handoff"] - mode_acc["junior_only"],
        "robustness_drop": mode_acc["tandem_handoff"] - mode_acc["corrupted_handoff"],
        "accuracy_by_task_type": _grouped_accuracy(results, "task_type"),
        "accuracy_by_difficulty": _grouped_accuracy(results, "difficulty"),
        "parse_status_counts": {
            str(key): int(value)
            for key, value in results["parse_status"].fillna("unknown").value_counts().sort_index().items()
        },
    }


def _row(
    task: Task,
    mode: str,
    senior_model: str,
    junior_model: str,
    response: AgentResponse,
    senior_reasoning: str = "",
    corrupted_reasoning: str = "",
) -> dict[str, Any]:
    correct = verify_final_answer(task, response.final_answer)
    parse_status = str(response.metadata.get("parse_status", "unknown"))
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "difficulty": task.difficulty,
        "mode": mode,
        "senior_model": senior_model,
        "junior_model": junior_model,
        "prompt": task.prompt,
        "expected_answer": task.answer,
        "model_answer": response.final_answer,
        "correct": correct,
        "senior_reasoning": senior_reasoning,
        "corrupted_reasoning": corrupted_reasoning,
        "raw_model_output": response.metadata.get("raw_output", ""),
        "parse_status": parse_status,
        "failure_type": _failure_type(correct, parse_status),
    }


def _produce_handoff(senior_agent: Agent, task: Task) -> AgentResponse:
    handoff_fn = getattr(senior_agent, "produce_handoff", None)
    if callable(handoff_fn):
        return handoff_fn(task)
    return senior_agent.answer(task)


def _failure_type(correct: bool, parse_status: str) -> str:
    if correct:
        return ""
    if parse_status == "parse_failed":
        return "parse_failed"
    return "incorrect"


def _accuracy_for(rows: pd.DataFrame) -> float:
    if len(rows) == 0:
        return 0.0
    return float(rows["correct"].mean())


def _grouped_accuracy(results: pd.DataFrame, group_column: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for group_value, group in results.groupby(group_column):
        grouped[str(group_value)] = {
            mode: _accuracy_for(group[group["mode"] == mode])
            for mode in ["senior_only", "junior_only", "tandem_handoff", "corrupted_handoff"]
        }
    return grouped


def _flat_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_tasks": summary["total_tasks"],
        "senior_only_accuracy": summary["senior_only_accuracy"],
        "junior_only_accuracy": summary["junior_only_accuracy"],
        "tandem_handoff_accuracy": summary["tandem_handoff_accuracy"],
        "corrupted_handoff_accuracy": summary["corrupted_handoff_accuracy"],
        "handoff_gain": summary["handoff_gain"],
        "robustness_drop": summary["robustness_drop"],
    }


if __name__ == "__main__":
    main()
