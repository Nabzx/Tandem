from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tandem_rlvr.agents import OracleSeniorAgent, WeakJuniorAgent
from tandem_rlvr.eval import EvaluationRunner
from tandem_rlvr.tasks import ArithmeticTaskGenerator
from tandem_rlvr.tasks.arithmetic import SUPPORTED_DIFFICULTIES, SUPPORTED_TASK_TYPES
from tandem_rlvr.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 1 TandemRLVR arithmetic baseline.")
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--difficulty", choices=SUPPORTED_DIFFICULTIES, default="mixed")
    parser.add_argument("--task-types", nargs="+", choices=SUPPORTED_TASK_TYPES, default=list(SUPPORTED_TASK_TYPES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=Path("outputs/results.csv"))
    args = parser.parse_args()

    seed_everything(args.seed)
    generator = ArithmeticTaskGenerator(
        seed=args.seed,
        task_types=args.task_types,
        difficulty=args.difficulty,
    )
    tasks = generator.generate(args.num_tasks)

    runner = EvaluationRunner(
        senior_agent=OracleSeniorAgent(),
        junior_agent=WeakJuniorAgent(),
        include_corrupted_handoff=False,
    )
    result = runner.run(tasks, output_path=args.output_path)

    summary_columns = [
        "num_tasks",
        "senior_only_correct",
        "senior_only_incorrect",
        "junior_only_correct",
        "junior_only_incorrect",
        "tandem_handoff_correct",
        "tandem_handoff_incorrect",
        "senior_only_accuracy",
        "junior_only_accuracy",
        "tandem_handoff_accuracy",
        "handoff_gain",
    ]
    summary = pd.DataFrame([{key: result.summary.as_dict()[key] for key in summary_columns}])
    print("\nStage 1 baseline summary")
    print(summary.to_string(index=False))
    print(f"\nWrote per-task results to {args.output_path}")


if __name__ == "__main__":
    main()
