from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from tandem_rlvr.agents.base import Agent
from tandem_rlvr.agents.llm import OllamaGenerationConfig, OllamaJuniorAgent, OllamaSeniorAgent
from tandem_rlvr.experiments.run_stage3_llm_eval import (
    ALL_MODES,
    parse_modes,
    resolve_generation_settings,
    run_stage3_llm_eval,
)
from tandem_rlvr.experiments.run_stage4_process_metrics import compute_stage4_metrics, summarize_stage4_metrics
from tandem_rlvr.tasks.splits import SUPPORTED_SPLITS, SUPPORTED_TASK_FAMILIES, build_split_benchmark
from tandem_rlvr.utils.io import ensure_parent_dir, write_json
from tandem_rlvr.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 5 generalization and OOD evaluation.")
    parser.add_argument("--num-tasks-per-split", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--senior-model", type=str, default="llama3.1:latest")
    parser.add_argument("--junior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--splits", type=str, default="id_eval,ood_eval,stress_eval")
    parser.add_argument("--task-families", type=str, default="arithmetic,list,logic,code")
    parser.add_argument("--modes", type=str, default="senior_only,junior_only,tandem_handoff,corrupted_handoff")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    num_predict, timeout_seconds = resolve_generation_settings(args)
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
    result = run_stage5_generalization_eval(
        num_tasks_per_split=args.num_tasks_per_split,
        seed=args.seed,
        senior_model=args.senior_model,
        junior_model=args.junior_model,
        splits=parse_splits(args.splits),
        task_families=parse_task_families(args.task_families),
        modes=parse_modes(args.modes),
        output_dir=args.output_dir,
        senior_agent=OllamaSeniorAgent(senior_config),
        junior_agent=OllamaJuniorAgent(junior_config),
        verbose=args.verbose,
    )

    print("\nStage 5 generalization summary")
    print_comparison_table(result["summary"])
    print(f"\nWrote Stage 5 results to {result['results_path']}")
    print(f"Wrote Stage 5 summary to {result['summary_path']}")
    print(f"Wrote Stage 5 process metrics to {result['process_metrics_path']}")
    print(f"Wrote Stage 5 process summary to {result['process_summary_path']}")


def run_stage5_generalization_eval(
    num_tasks_per_split: int,
    seed: int,
    senior_model: str,
    junior_model: str,
    splits: list[str],
    task_families: list[str],
    modes: list[str],
    output_dir: str | Path,
    senior_agent: Agent,
    junior_agent: Agent,
    verbose: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tasks = build_split_benchmark(
        num_tasks_per_split=num_tasks_per_split,
        splits=splits,
        seed=seed,
        task_families=task_families,
    )
    results_path = output_dir / "stage5_generalization_results.csv"
    summary_path = output_dir / "stage5_generalization_summary.json"
    process_metrics_path = output_dir / "stage5_process_metrics.csv"
    process_summary_path = output_dir / "stage5_process_summary.json"

    stage3_result = run_stage3_llm_eval(
        tasks=tasks,
        senior_agent=senior_agent,
        junior_agent=junior_agent,
        senior_model=senior_model,
        junior_model=junior_model,
        seed=seed,
        results_path=results_path,
        summary_path=output_dir / "stage5_stage3_summary.json",
        modes=modes,
        verbose=verbose,
    )
    results = stage3_result["results"]
    ensure_parent_dir(results_path)
    results.to_csv(results_path, index=False)

    process_metrics = compute_stage4_metrics(results)
    ensure_parent_dir(process_metrics_path)
    process_metrics.to_csv(process_metrics_path, index=False)
    process_summary = summarize_stage4_metrics(process_metrics)
    write_json(process_summary, process_summary_path)

    summary = summarize_stage5_results(results, process_metrics, splits)
    write_json(summary, summary_path)
    return {
        "results": results,
        "process_metrics": process_metrics,
        "summary": summary,
        "process_summary": process_summary,
        "results_path": results_path,
        "summary_path": summary_path,
        "process_metrics_path": process_metrics_path,
        "process_summary_path": process_summary_path,
    }


def summarize_stage5_results(results: pd.DataFrame, process_metrics: pd.DataFrame, splits: list[str]) -> dict[str, Any]:
    accuracy_by_split_and_mode = {
        split: {
            mode: _accuracy(results[(results["split"] == split) & (results["mode"] == mode)])
            for mode in ALL_MODES
        }
        for split in splits
    }
    summary = {
        "accuracy_by_split": {
            split: _accuracy(results[results["split"] == split])
            for split in splits
        },
        "accuracy_by_split_and_mode": accuracy_by_split_and_mode,
        "accuracy_by_split_and_task_type": _accuracy_by_split_and_task_type(results, splits),
        "handoff_gain_by_split": {
            split: _difference_if_valid(
                accuracy_by_split_and_mode[split]["tandem_handoff"],
                accuracy_by_split_and_mode[split]["junior_only"],
            )
            for split in splits
        },
        "robustness_drop_by_split": {
            split: _difference_if_valid(
                accuracy_by_split_and_mode[split]["tandem_handoff"],
                accuracy_by_split_and_mode[split]["corrupted_handoff"],
            )
            for split in splits
        },
        "process_reward_by_split": _process_mean_by_split(process_metrics, "process_reward_score", splits),
        "legibility_by_split": _process_mean_by_split(process_metrics, "legibility_score", splits),
        "leakage_by_split": _process_mean_by_split(process_metrics, "leakage_score", splits),
        "relevance_by_split": _process_mean_by_split(process_metrics, "relevance_score", splits),
        "usefulness_by_split": _process_mean_by_split(process_metrics, "usefulness_score", splits),
        "failure_type_counts_by_split": _counts_by_split(results, "failure_type", splits),
        "parse_status_counts_by_split": _counts_by_split(results, "parse_status", splits),
    }
    id_tandem = accuracy_by_split_and_mode.get("id_eval", {}).get("tandem_handoff")
    ood_tandem = accuracy_by_split_and_mode.get("ood_eval", {}).get("tandem_handoff")
    stress_tandem = accuracy_by_split_and_mode.get("stress_eval", {}).get("tandem_handoff")
    summary["ood_generalization_gap"] = _difference_if_valid(id_tandem, ood_tandem)
    summary["stress_generalization_gap"] = _difference_if_valid(id_tandem, stress_tandem)
    return summary


def print_comparison_table(summary: dict[str, Any]) -> None:
    rows = []
    for split, mode_acc in summary["accuracy_by_split_and_mode"].items():
        rows.append(
            {
                "split": split,
                "senior_only": mode_acc.get("senior_only"),
                "junior_only": mode_acc.get("junior_only"),
                "tandem": mode_acc.get("tandem_handoff"),
                "corrupted": mode_acc.get("corrupted_handoff"),
                "handoff_gain": summary["handoff_gain_by_split"].get(split),
                "process_reward": summary["process_reward_by_split"].get(split),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def parse_splits(raw_splits: str) -> list[str]:
    splits = [split.strip() for split in raw_splits.split(",") if split.strip()]
    unknown = set(splits) - set(SUPPORTED_SPLITS)
    if unknown:
        raise SystemExit(f"Unsupported splits: {sorted(unknown)}")
    if not splits:
        raise SystemExit("At least one split must be selected.")
    return splits


def parse_task_families(raw_families: str) -> list[str]:
    families = [family.strip() for family in raw_families.split(",") if family.strip()]
    unknown = set(families) - set(SUPPORTED_TASK_FAMILIES)
    if unknown:
        raise SystemExit(f"Unsupported task families: {sorted(unknown)}")
    if not families:
        raise SystemExit("At least one task family must be selected.")
    return families


def _accuracy(rows: pd.DataFrame) -> float | None:
    if rows.empty:
        return None
    return float(rows["correct"].mean())


def _difference_if_valid(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _accuracy_by_split_and_task_type(results: pd.DataFrame, splits: list[str]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for split in splits:
        split_rows = results[results["split"] == split]
        output[split] = {
            str(task_type): _accuracy(task_rows)
            for task_type, task_rows in split_rows.groupby("task_type")
        }
    return output


def _process_mean_by_split(process_metrics: pd.DataFrame, column: str, splits: list[str]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for split in splits:
        rows = process_metrics[process_metrics["split"] == split] if "split" in process_metrics else pd.DataFrame()
        values = pd.to_numeric(rows[column], errors="coerce").dropna() if column in rows else pd.Series(dtype=float)
        output[split] = None if values.empty else float(values.mean())
    return output


def _counts_by_split(results: pd.DataFrame, column: str, splits: list[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split in splits:
        rows = results[results["split"] == split]
        output[split] = {
            str(key): int(value)
            for key, value in rows[column].fillna("unknown").value_counts().sort_index().items()
            if str(key) != ""
        }
    return output


if __name__ == "__main__":
    main()
