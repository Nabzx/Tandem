from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tandem_rlvr.agents import OracleSeniorAgent, WeakJuniorAgent
from tandem_rlvr.eval import EvaluationRunner
from tandem_rlvr.tasks import (
    ArithmeticTaskGenerator,
    CodeTracingTaskGenerator,
    ListTransformationTaskGenerator,
    LogicTaskGenerator,
    Task,
)
from tandem_rlvr.utils.io import write_json
from tandem_rlvr.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 TandemRLVR mixed-task baseline.")
    parser.add_argument("--num-tasks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    parser.add_argument("--results-path", type=Path, default=Path("outputs/stage2_results.csv"))
    parser.add_argument("--summary-path", type=Path, default=Path("outputs/stage2_summary.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    tasks = build_mixed_benchmark(args.num_tasks, seed=args.seed, difficulty=args.difficulty)

    runner = EvaluationRunner(
        senior_agent=OracleSeniorAgent(),
        junior_agent=WeakJuniorAgent(),
        seed=args.seed,
    )
    result = runner.run(tasks, output_path=args.results_path)
    write_json(result.summary.as_dict(), args.summary_path)

    summary = result.summary.as_dict()
    summary_table = pd.DataFrame(
        [
            {
                "num_tasks": summary["num_tasks"],
                "senior_only_accuracy": summary["senior_only_accuracy"],
                "junior_only_accuracy": summary["junior_only_accuracy"],
                "tandem_handoff_accuracy": summary["tandem_handoff_accuracy"],
                "corrupted_handoff_accuracy": summary["corrupted_handoff_accuracy"],
                "handoff_gain": summary["handoff_gain"],
                "robustness_drop": summary["robustness_drop"],
            }
        ]
    )
    print("\nStage 2 baseline summary")
    print(summary_table.to_string(index=False))
    print("\nTask type counts")
    print(pd.Series(summary["task_type_counts"]).sort_index().to_string())
    print(f"\nWrote per-task results to {args.results_path}")
    print(f"Wrote summary metrics to {args.summary_path}")


def build_mixed_benchmark(num_tasks: int, seed: int, difficulty: str = "mixed") -> list[Task]:
    if num_tasks < 0:
        raise ValueError("num_tasks must be non-negative")

    generators = [
        ArithmeticTaskGenerator(seed=seed + 11, difficulty=difficulty),
        ListTransformationTaskGenerator(seed=seed + 23, difficulty=difficulty),
        LogicTaskGenerator(seed=seed + 37, difficulty=difficulty),
        CodeTracingTaskGenerator(seed=seed + 51, difficulty=difficulty),
    ]
    tasks: list[Task] = []
    for index in range(num_tasks):
        tasks.append(generators[index % len(generators)].generate_one())
    return tasks


if __name__ == "__main__":
    main()
