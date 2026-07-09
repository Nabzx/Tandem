from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tandem_rlvr.agents.base import Agent, AgentResponse
from tandem_rlvr.agents.llm import (
    OllamaBackendUnavailable,
    OllamaGenerationConfig,
    OllamaJuniorAgent,
    OllamaSeniorAgent,
)
from tandem_rlvr.agents.llm.prompts import junior_direct_prompt, junior_tandem_prompt, senior_direct_prompt, senior_handoff_prompt
from tandem_rlvr.eval.perturbations import light_noise
from tandem_rlvr.experiments.run_stage2_baseline import build_mixed_benchmark
from tandem_rlvr.tasks import ArithmeticTaskGenerator, CodeTracingTaskGenerator, ListTransformationTaskGenerator, LogicTaskGenerator, Task
from tandem_rlvr.tasks.verifiers import verify_final_answer
from tandem_rlvr.utils.io import ensure_parent_dir, write_json
from tandem_rlvr.utils.seed import seed_everything


MODE_ALIASES = {
    "senior_only": "senior_only",
    "junior_only": "junior_only",
    "tandem": "tandem_handoff",
    "tandem_handoff": "tandem_handoff",
    "corrupted": "corrupted_handoff",
    "corrupted_handoff": "corrupted_handoff",
}
ALL_MODES = ("senior_only", "junior_only", "tandem_handoff", "corrupted_handoff")
TASK_FAMILIES = ("arithmetic", "list", "logic", "code", "all")
NORMAL_NUM_PREDICT = 256
QUICK_NUM_PREDICT = 128
NORMAL_TIMEOUT_SECONDS = 60
QUICK_TIMEOUT_SECONDS = 30
DEFAULT_TEMPERATURE = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 TandemRLVR evaluation with local Ollama models.")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--senior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--junior-model", type=str, default="llama3.2:1b")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--quick", action="store_true", help="Use shorter generations for faster local smoke tests.")
    parser.add_argument("--verbose", action="store_true", help="Print prompts, raw outputs, parsing, answers, and correctness.")
    parser.add_argument("--debug-save-prompts", action="store_true", help="Save exact prompts sent to Ollama under outputs/debug_prompts/.")
    parser.add_argument("--task-types", type=str, default="all", help="Comma-separated task families: arithmetic,list,logic,code,all.")
    parser.add_argument("--easy-only", action="store_true", help="Generate only easy tasks.")
    parser.add_argument(
        "--modes",
        type=str,
        default="senior_only,junior_only,tandem,corrupted",
        help="Comma-separated modes: senior_only,junior_only,tandem,corrupted.",
    )
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
    modes = parse_modes(args.modes)
    task_families = parse_task_families(args.task_types)
    num_predict, timeout_seconds = resolve_generation_settings(args)
    _print_start(args, modes, task_families, num_predict, timeout_seconds)
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
    difficulty = "easy" if args.easy_only else args.difficulty
    tasks = build_stage3_benchmark(args.num_tasks, seed=args.seed, difficulty=difficulty, task_families=task_families)
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
        modes=modes,
        verbose=args.verbose,
        debug_prompts_dir=output_dir / "debug_prompts" if args.debug_save_prompts else None,
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
    modes: list[str] | None = None,
    verbose: bool = False,
    debug_prompts_dir: str | Path | None = None,
) -> dict[str, Any]:
    import random

    modes = modes or list(ALL_MODES)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    debug_prompts_path = Path(debug_prompts_dir) if debug_prompts_dir is not None else None
    print("Starting Stage 3 local LLM evaluation", flush=True)
    print(f"Senior model: {senior_model}", flush=True)
    print(f"Junior model: {junior_model}", flush=True)
    print(f"Number of tasks: {len(tasks)}", flush=True)
    print(f"Modes: {', '.join(modes)}", flush=True)
    for task_index, task in enumerate(tasks, start=1):
        senior_handoff: AgentResponse | None = None
        senior_handoff_generation_seconds: float | None = None
        if "senior_only" in modes:
            _print_progress(task_index, len(tasks), task, "senior_only")
            prompt = senior_direct_prompt(task)
            _save_prompt(debug_prompts_path, task, "senior_only", senior_model, prompt)
            _print_verbose_prompt(verbose, prompt)
            senior_only, _ = _timed_agent_call(lambda: senior_agent.answer(task))
            row = _row(task, "senior_only", senior_model, junior_model, senior_only, senior_reasoning=senior_only.reasoning)
            rows.append(row)
            _print_verbose_result(verbose, row)

        if "junior_only" in modes:
            _print_progress(task_index, len(tasks), task, "junior_only")
            prompt = junior_direct_prompt(task)
            _save_prompt(debug_prompts_path, task, "junior_only", junior_model, prompt)
            _print_verbose_prompt(verbose, prompt)
            junior_only, _ = _timed_agent_call(lambda: junior_agent.answer(task))
            row = _row(task, "junior_only", senior_model, junior_model, junior_only)
            rows.append(row)
            _print_verbose_result(verbose, row)

        if "tandem_handoff" in modes or "corrupted_handoff" in modes:
            requested_handoff_modes = [mode for mode in ("tandem_handoff", "corrupted_handoff") if mode in modes]
            print(
                f"[{task_index}/{len(tasks)}] generating senior handoff reasoning for {','.join(requested_handoff_modes)}",
                flush=True,
            )
            prompt = senior_handoff_prompt(task)
            _save_prompt(debug_prompts_path, task, "senior_handoff", senior_model, prompt)
            _print_verbose_prompt(verbose, prompt)
            senior_handoff, senior_handoff_generation_seconds = _timed_agent_call(lambda: _produce_handoff(senior_agent, task))
            _print_verbose_internal_handoff(verbose, senior_handoff)

        if "tandem_handoff" in modes and senior_handoff is not None:
            print(f"[{task_index}/{len(tasks)}] scoring junior tandem_handoff answer", flush=True)
            prompt = junior_tandem_prompt(task, senior_handoff.reasoning)
            _save_prompt(debug_prompts_path, task, "tandem_handoff", junior_model, prompt)
            _print_verbose_prompt(verbose, prompt)
            tandem, _ = _timed_agent_call(lambda: junior_agent.answer(task, context=senior_handoff.reasoning))
            row = _row(
                task,
                "tandem_handoff",
                senior_model,
                junior_model,
                tandem,
                senior_reasoning=senior_handoff.reasoning,
                senior_handoff=senior_handoff,
                senior_handoff_generation_seconds=senior_handoff_generation_seconds,
            )
            rows.append(row)
            _print_verbose_result(verbose, row)

        if "corrupted_handoff" in modes and senior_handoff is not None:
            corrupted_reasoning = light_noise(senior_handoff.reasoning, rng=rng)
            print(f"[{task_index}/{len(tasks)}] scoring junior corrupted_handoff answer", flush=True)
            prompt = junior_tandem_prompt(task, corrupted_reasoning)
            _save_prompt(debug_prompts_path, task, "corrupted_handoff", junior_model, prompt)
            _print_verbose_prompt(verbose, prompt)
            corrupted, _ = _timed_agent_call(lambda: junior_agent.answer(task, context=corrupted_reasoning))
            row = _row(
                task,
                "corrupted_handoff",
                senior_model,
                junior_model,
                corrupted,
                senior_reasoning=senior_handoff.reasoning,
                corrupted_reasoning=corrupted_reasoning,
                senior_handoff=senior_handoff,
                senior_handoff_generation_seconds=senior_handoff_generation_seconds,
            )
            rows.append(row)
            _print_verbose_result(verbose, row)

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
        for mode in ALL_MODES
    }
    task_level = results.drop_duplicates("task_id")
    handoff_gain = _difference_if_valid(mode_acc["tandem_handoff"], mode_acc["junior_only"])
    robustness_drop = _difference_if_valid(mode_acc["tandem_handoff"], mode_acc["corrupted_handoff"])
    return {
        "total_tasks": int(len(task_level)),
        "senior_model": senior_model,
        "junior_model": junior_model,
        "senior_only_accuracy": mode_acc["senior_only"],
        "junior_only_accuracy": mode_acc["junior_only"],
        "tandem_handoff_accuracy": mode_acc["tandem_handoff"],
        "corrupted_handoff_accuracy": mode_acc["corrupted_handoff"],
        "handoff_gain": handoff_gain,
        "robustness_drop": robustness_drop,
        "accuracy_by_task_type": _grouped_accuracy(results, "task_type"),
        "accuracy_by_mode_and_task_type": _accuracy_by_mode_and_task_type(results),
        "accuracy_by_difficulty": _grouped_accuracy(results, "difficulty"),
        "parse_status_counts": {
            str(key): int(value)
            for key, value in results["parse_status"].fillna("unknown").value_counts().sort_index().items()
        },
        "parse_status_counts_by_mode": _counts_by_mode(results, "parse_status"),
        "failure_type_counts_by_mode": _counts_by_mode(results, "failure_type"),
    }


def _row(
    task: Task,
    mode: str,
    senior_model: str,
    junior_model: str,
    response: AgentResponse,
    senior_reasoning: str = "",
    corrupted_reasoning: str = "",
    senior_handoff: AgentResponse | None = None,
    senior_handoff_generation_seconds: float | None = None,
) -> dict[str, Any]:
    raw_model_answer = "" if response.final_answer is None else str(response.final_answer)
    normalized_expected_answer = _normalize_answer_for_diagnostics(task, task.answer)
    normalized_model_answer = _normalize_answer_for_diagnostics(task, raw_model_answer)
    parse_status = str(response.metadata.get("parse_status", "unknown"))
    correct, verifier_error = _safe_verify(task, normalized_model_answer)
    failure_type = _failure_type(
        correct=correct,
        parse_status=parse_status,
        raw_output=str(response.metadata.get("raw_output", "")),
        raw_model_answer=raw_model_answer,
        verifier_error=verifier_error,
        metadata=response.metadata,
    )
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "task_family": task.metadata.get("task_family", _infer_task_family(task.task_type)),
        "difficulty": task.difficulty,
        "split": task.metadata.get("split", ""),
        "distribution": task.metadata.get("distribution", ""),
        "ood_type": task.metadata.get("ood_type", ""),
        "mode": mode,
        "senior_model": senior_model,
        "junior_model": junior_model,
        "prompt": task.prompt,
        "expected_answer": task.answer,
        "model_answer": normalized_model_answer,
        "raw_expected_answer": task.answer,
        "normalized_expected_answer": normalized_expected_answer,
        "raw_model_answer": raw_model_answer,
        "normalized_model_answer": normalized_model_answer,
        "correct": correct,
        "senior_reasoning": senior_reasoning,
        "senior_handoff_reasoning": "" if senior_handoff is None else senior_handoff.reasoning,
        "senior_handoff_raw_output": "" if senior_handoff is None else senior_handoff.metadata.get("raw_output", ""),
        "senior_handoff_parse_status": "" if senior_handoff is None else senior_handoff.metadata.get("parse_status", ""),
        "senior_handoff_generation_seconds": senior_handoff_generation_seconds,
        "corrupted_reasoning": corrupted_reasoning,
        "raw_model_output": response.metadata.get("raw_output", ""),
        "parse_status": parse_status,
        "failure_type": failure_type,
    }


def _produce_handoff(senior_agent: Agent, task: Task) -> AgentResponse:
    handoff_fn = getattr(senior_agent, "produce_handoff", None)
    if callable(handoff_fn):
        return handoff_fn(task)
    return senior_agent.answer(task)


def _failure_type(
    correct: bool,
    parse_status: str,
    raw_output: str,
    raw_model_answer: str,
    verifier_error: bool,
    metadata: dict[str, Any],
) -> str:
    if correct:
        return ""
    if metadata.get("failure_type") == "timeout":
        return "timeout"
    if metadata.get("failure_type") == "backend_error":
        return "backend_error"
    if verifier_error:
        return "verifier_error"
    if parse_status == "parse_failed":
        if _looks_truncated_json(raw_output):
            return "truncated_json"
        return "parse_failed"
    if raw_model_answer.strip() == "":
        return "empty_answer"
    return "wrong_answer"


def _accuracy_for(rows: pd.DataFrame) -> float | None:
    if len(rows) == 0:
        return None
    return float(rows["correct"].mean())


def _difference_if_valid(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _grouped_accuracy(results: pd.DataFrame, group_column: str) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, dict[str, float | None]] = {}
    for group_value, group in results.groupby(group_column):
        grouped[str(group_value)] = {
            mode: _accuracy_for(group[group["mode"] == mode])
            for mode in ALL_MODES
        }
    return grouped


def _accuracy_by_mode_and_task_type(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for mode in ALL_MODES:
        mode_rows = results[results["mode"] == mode]
        grouped[mode] = {
            str(task_type): _accuracy_for(task_rows)
            for task_type, task_rows in mode_rows.groupby("task_type")
        }
    return grouped


def _counts_by_mode(results: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for mode in ALL_MODES:
        mode_rows = results[results["mode"] == mode]
        counts[mode] = {
            str(key): int(value)
            for key, value in mode_rows[column].fillna("unknown").value_counts().sort_index().items()
            if str(key) != ""
        }
    return counts


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


def parse_modes(raw_modes: str) -> list[str]:
    modes: list[str] = []
    for raw_mode in raw_modes.split(","):
        key = raw_mode.strip()
        if not key:
            continue
        if key not in MODE_ALIASES:
            valid = ", ".join(MODE_ALIASES)
            raise SystemExit(f"Unknown mode '{key}'. Valid modes: {valid}")
        mode = MODE_ALIASES[key]
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise SystemExit("At least one mode must be selected.")
    return modes


def parse_task_families(raw_task_types: str) -> list[str]:
    families: list[str] = []
    for raw_family in raw_task_types.split(","):
        family = raw_family.strip()
        if not family:
            continue
        if family not in TASK_FAMILIES:
            valid = ", ".join(TASK_FAMILIES)
            raise SystemExit(f"Unknown task family '{family}'. Valid task families: {valid}")
        if family == "all":
            return ["arithmetic", "list", "logic", "code"]
        if family not in families:
            families.append(family)
    if not families:
        raise SystemExit("At least one task family must be selected.")
    return families


def build_stage3_benchmark(num_tasks: int, seed: int, difficulty: str, task_families: list[str]) -> list[Task]:
    if task_families == ["arithmetic", "list", "logic", "code"]:
        return build_mixed_benchmark(num_tasks, seed=seed, difficulty=difficulty)
    generators = []
    if "arithmetic" in task_families:
        generators.append(ArithmeticTaskGenerator(seed=seed + 11, difficulty=difficulty))
    if "list" in task_families:
        generators.append(ListTransformationTaskGenerator(seed=seed + 23, difficulty=difficulty))
    if "logic" in task_families:
        generators.append(LogicTaskGenerator(seed=seed + 37, difficulty=difficulty))
    if "code" in task_families:
        generators.append(CodeTracingTaskGenerator(seed=seed + 51, difficulty=difficulty))
    tasks: list[Task] = []
    for index in range(num_tasks):
        tasks.append(generators[index % len(generators)].generate_one())
    return tasks


def resolve_generation_settings(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_predict is not None:
        num_predict = args.num_predict
    elif args.quick:
        num_predict = QUICK_NUM_PREDICT
    else:
        num_predict = NORMAL_NUM_PREDICT

    if args.timeout_seconds is not None:
        timeout_seconds = args.timeout_seconds
    elif args.max_generation_seconds is not None:
        timeout_seconds = args.max_generation_seconds
    elif args.quick:
        timeout_seconds = QUICK_TIMEOUT_SECONDS
    else:
        timeout_seconds = NORMAL_TIMEOUT_SECONDS

    return num_predict, timeout_seconds


def _print_start(args: argparse.Namespace, modes: list[str], task_families: list[str], num_predict: int, timeout_seconds: int) -> None:
    print("Preparing Stage 3 local LLM evaluation", flush=True)
    print(f"Senior model: {args.senior_model}", flush=True)
    print(f"Junior model: {args.junior_model}", flush=True)
    print(f"Number of tasks: {args.num_tasks}", flush=True)
    print(f"Modes: {', '.join(modes)}", flush=True)
    print(f"Task families: {', '.join(task_families)}", flush=True)
    print(f"Difficulty: {'easy' if args.easy_only else args.difficulty}", flush=True)
    print(f"num_predict: {num_predict}", flush=True)
    print(f"max generation seconds: {timeout_seconds}", flush=True)


def _print_progress(task_index: int, total_tasks: int, task: Task, mode: str) -> None:
    print(f"[{task_index}/{total_tasks}] task_type={task.task_type} mode={mode}", flush=True)


def _print_verbose_prompt(verbose: bool, prompt: str) -> None:
    if verbose:
        print(f"  prompt preview: {_preview(prompt)}", flush=True)


def _print_verbose_result(verbose: bool, row: dict[str, Any]) -> None:
    if not verbose:
        return
    print(f"  raw output preview: {_preview(str(row['raw_model_output']))}", flush=True)
    print(f"  parse status: {row['parse_status']}", flush=True)
    print(f"  normalized answer: {row['normalized_model_answer']}", flush=True)
    print(f"  correct: {row['correct']}", flush=True)


def _print_verbose_internal_handoff(verbose: bool, response: AgentResponse) -> None:
    if not verbose:
        return
    print("  mode=senior_handoff_internal (not scored)", flush=True)
    print(f"  raw output preview: {_preview(str(response.metadata.get('raw_output', '')))}", flush=True)
    print(f"  parse status: {response.metadata.get('parse_status', 'unknown')}", flush=True)


def _preview(text: str, max_chars: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _save_prompt(debug_prompts_dir: Path | None, task: Task, mode: str, model_name: str, prompt: str) -> None:
    if debug_prompts_dir is None:
        return
    debug_prompts_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_filename(task.task_id)}_{_safe_filename(mode)}_{_safe_filename(model_name)}.txt"
    (debug_prompts_dir / filename).write_text(prompt, encoding="utf-8")


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _timed_agent_call(call: Callable[[], AgentResponse]) -> tuple[AgentResponse, float]:
    start = time.monotonic()
    response = _safe_agent_call(call)
    return response, time.monotonic() - start


def _safe_agent_call(call: Callable[[], AgentResponse]) -> AgentResponse:
    try:
        return call()
    except TimeoutError as exc:
        return _error_response("timeout", exc)
    except Exception as exc:
        message = str(exc).lower()
        if "timeout" in message or "timed out" in message:
            return _error_response("timeout", exc)
        return _error_response("backend_error", exc)


def _error_response(failure_type: str, exc: Exception) -> AgentResponse:
    return AgentResponse(
        reasoning="",
        final_answer="",
        metadata={
            "raw_output": "",
            "parse_status": "parse_failed",
            "failure_type": failure_type,
            "error": str(exc),
        },
    )


def _normalize_answer_for_diagnostics(task: Task, answer: str) -> str:
    from tandem_rlvr.agents.llm.parsing import normalize_llm_answer

    return normalize_llm_answer(task, answer)


def _safe_verify(task: Task, normalized_model_answer: str) -> tuple[bool, bool]:
    try:
        return verify_final_answer(task, normalized_model_answer), False
    except Exception:
        return False, True


def _looks_truncated_json(raw_output: str) -> bool:
    stripped = raw_output.strip()
    return "{" in stripped and not stripped.endswith("}")


def _infer_task_family(task_type: str) -> str:
    if task_type in {"addition", "subtraction", "multiplication", "arithmetic_two_step"}:
        return "arithmetic"
    if task_type.startswith("list_"):
        return "list"
    if task_type.startswith("logic_"):
        return "logic"
    if task_type.startswith("code_trace_"):
        return "code"
    return "unknown"


if __name__ == "__main__":
    main()
