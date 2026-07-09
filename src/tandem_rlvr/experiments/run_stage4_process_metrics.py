from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import pandas as pd

from tandem_rlvr.metrics import (
    compute_legibility_metrics,
    compute_leakage_metrics,
    compute_process_reward,
    compute_relevance_metrics,
    compute_usefulness_metrics_by_task,
)
from tandem_rlvr.tasks.base import Task
from tandem_rlvr.utils.io import ensure_parent_dir, write_json


HANDOFF_MODES = {"tandem_handoff", "corrupted_handoff"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deterministic Stage 4 process metrics from Stage 3 results.")
    parser.add_argument("--input", type=Path, default=Path("outputs/stage3_llm_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    result = run_stage4_process_metrics(args.input, args.output_dir)
    print("\nStage 4 process metrics summary")
    print(pd.DataFrame([_flat_summary(result["summary"])]).to_string(index=False))
    print(f"\nWrote per-row process metrics to {result['metrics_path']}")
    print(f"Wrote summary metrics to {result['summary_path']}")


def run_stage4_process_metrics(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    results = pd.read_csv(input_path)
    metrics = compute_stage4_metrics(results)

    metrics_path = output_dir / "stage4_process_metrics.csv"
    summary_path = output_dir / "stage4_process_summary.json"
    ensure_parent_dir(metrics_path)
    metrics.to_csv(metrics_path, index=False)
    summary = summarize_stage4_metrics(metrics)
    write_json(summary, summary_path)
    return {"metrics": metrics, "summary": summary, "metrics_path": metrics_path, "summary_path": summary_path}


def compute_stage4_metrics(stage3_results: pd.DataFrame) -> pd.DataFrame:
    usefulness_by_task = compute_usefulness_metrics_by_task(stage3_results)
    rows: list[dict[str, Any]] = []
    handoff_rows = stage3_results[stage3_results["mode"].isin(HANDOFF_MODES)]
    for _, row in handoff_rows.iterrows():
        task = _task_from_row(row)
        reasoning = _effective_reasoning(row)
        usefulness = usefulness_by_task.get(str(row["task_id"]), {})
        legibility = compute_legibility_metrics(task, reasoning)
        leakage = compute_leakage_metrics(task, reasoning, expected_answer=str(row.get("raw_expected_answer", row.get("expected_answer", ""))))
        relevance = compute_relevance_metrics(task, reasoning)
        components = {
            **legibility,
            **leakage,
            **relevance,
            "usefulness_score": usefulness.get("usefulness_score"),
        }
        process = compute_process_reward(components)
        rows.append(
            {
                "task_id": row["task_id"],
                "task_type": row["task_type"],
                "difficulty": row["difficulty"],
                "mode": row["mode"],
                "expected_answer": row.get("raw_expected_answer", row.get("expected_answer", "")),
                "model_answer": row.get("raw_model_answer", row.get("model_answer", "")),
                "correct": bool(row.get("correct", False)),
                "reasoning_evaluated": reasoning,
                "reasoning_source": "corrupted_reasoning" if row["mode"] == "corrupted_handoff" else "senior_handoff_reasoning",
                **legibility,
                **leakage,
                **relevance,
                **usefulness,
                **process,
            }
        )
    return pd.DataFrame(rows)


def summarize_stage4_metrics(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "num_rows_scored": 0,
            "num_tasks": 0,
            "mean_legibility_score": None,
            "mean_leakage_score": None,
            "mean_relevance_score": None,
            "mean_usefulness_score": None,
            "mean_process_reward_score": None,
            "leakage_rate": None,
            "hallucination_flag_rate": None,
            "handoff_improvement_rate": None,
            "handoff_harm_rate": None,
            "corruption_sensitivity_rate": None,
            "mean_reasoning_word_count": None,
            "breakdowns": {},
        }

    return {
        "mean_legibility_score": _mean(metrics, "legibility_score"),
        "mean_leakage_score": _mean(metrics, "leakage_score"),
        "mean_relevance_score": _mean(metrics, "relevance_score"),
        "mean_usefulness_score": _mean(metrics, "usefulness_score"),
        "mean_process_reward_score": _mean(metrics, "process_reward_score"),
        "leakage_rate": _rate(metrics["leaks_exact_answer"] | metrics["leaks_normalized_answer"]),
        "hallucination_flag_rate": _rate(metrics["hallucination_flags"].apply(_has_flags)),
        "handoff_improvement_rate": _rate(metrics.drop_duplicates("task_id")["handoff_improved"]),
        "handoff_harm_rate": _rate(metrics.drop_duplicates("task_id")["handoff_hurt"]),
        "corruption_sensitivity_rate": _rate(metrics.drop_duplicates("task_id")["corruption_hurt"]),
        "mean_reasoning_word_count": _mean(metrics, "reasoning_word_count"),
        "num_rows_scored": int(len(metrics)),
        "num_tasks": int(metrics["task_id"].nunique()),
        "breakdowns": {
            "task_type": _breakdown(metrics, "task_type"),
            "difficulty": _breakdown(metrics, "difficulty"),
            "mode": _breakdown(metrics, "mode"),
        },
    }


def _task_from_row(row: pd.Series) -> Task:
    answer = str(row.get("raw_expected_answer", row.get("expected_answer", "")))
    metadata = {"answer_type": _infer_answer_type(answer)}
    return Task(
        task_id=str(row["task_id"]),
        task_type=str(row["task_type"]),
        prompt=str(row.get("prompt", "")),
        answer=answer,
        difficulty=str(row["difficulty"]),
        metadata=metadata,
    )


def _effective_reasoning(row: pd.Series) -> str:
    if row["mode"] == "corrupted_handoff":
        corrupted = row.get("corrupted_reasoning", "")
        if isinstance(corrupted, str) and corrupted.strip():
            return corrupted
    for column in ["senior_handoff_reasoning", "senior_reasoning"]:
        value = row.get(column, "")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _infer_answer_type(answer: str) -> str:
    stripped = answer.strip().lower()
    if stripped in {"true", "false"}:
        return "bool"
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return "list_int"
    try:
        int(stripped)
    except ValueError:
        return "text"
    return "int"


def _mean(metrics: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(metrics[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _rate(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.fillna(False).astype(bool).mean())


def _has_flags(value: Any) -> bool:
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        stripped = value.strip()
        return stripped not in {"", "[]"}
    return False


def _breakdown(metrics: pd.DataFrame, group_column: str) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for group_value, group in metrics.groupby(group_column):
        breakdown[str(group_value)] = {
            "num_rows_scored": int(len(group)),
            "mean_legibility_score": _mean(group, "legibility_score"),
            "mean_leakage_score": _mean(group, "leakage_score"),
            "mean_relevance_score": _mean(group, "relevance_score"),
            "mean_usefulness_score": _mean(group, "usefulness_score"),
            "mean_process_reward_score": _mean(group, "process_reward_score"),
            "leakage_rate": _rate(group["leaks_exact_answer"] | group["leaks_normalized_answer"]),
            "hallucination_flag_rate": _rate(group["hallucination_flags"].apply(_has_flags)),
            "mean_reasoning_word_count": _mean(group, "reasoning_word_count"),
        }
    return breakdown


def _flat_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_rows_scored": summary["num_rows_scored"],
        "num_tasks": summary["num_tasks"],
        "mean_legibility_score": summary["mean_legibility_score"],
        "mean_leakage_score": summary["mean_leakage_score"],
        "mean_relevance_score": summary["mean_relevance_score"],
        "mean_usefulness_score": summary["mean_usefulness_score"],
        "mean_process_reward_score": summary["mean_process_reward_score"],
    }


if __name__ == "__main__":
    main()
