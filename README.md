# TandemRLVR

TandemRLVR is a small research infrastructure project for studying **Tandem Reinforcement Learning with Verifiable Rewards**: evaluating whether a stronger "senior" reasoning agent can produce intermediate reasoning that remains useful to a weaker "junior" agent.

This repository currently implements **Stages 1 and 2**: a minimal, testable evaluation scaffold with synthetic tasks, tandem handoff evaluation, and corrupted-handoff robustness tests. It does not implement RL training, PPO/GRPO, learned rewards, or real language-model agents yet.

## Research Motivation

Outcome-only RLVR can reward models for reaching correct final answers, but final-answer correctness does not guarantee that intermediate reasoning is legible, inspectable, or useful for oversight. TandemRLVR asks whether we can evaluate and eventually train agents so that a weaker overseer can continue from a stronger agent's partial reasoning.

The Stage 1 tandem setup is:

1. A senior agent receives a task and produces reasoning.
2. A junior agent receives the task and the senior's reasoning, but not the senior's final answer.
3. The junior must produce the final answer.

If the junior succeeds more often with the senior's reasoning than without it, the handoff may be useful. In later stages, this can become a proxy for weak-overseer legibility and robustness.

## Stage 1 Scope

Implemented:

- Synthetic arithmetic task generation for addition, subtraction, and multiplication.
- Clean task and agent data structures.
- Programmatic final-answer verification.
- Placeholder agents:
  - `OracleSeniorAgent`
  - `WeakJuniorAgent`
  - `RandomAgent`
- Evaluation modes:
  - senior-only
  - junior-only
  - tandem handoff
- Metrics:
  - `senior_only_accuracy`
  - `junior_only_accuracy`
  - `tandem_handoff_accuracy`
  - `handoff_gain`
  - correct/incorrect counts per mode
- CSV output at `outputs/results.csv`.
- Pytest coverage for generation, verification, agents, and evaluation.

Not implemented yet:

- Real LLM wrappers.
- Hugging Face or API model integrations.
- RL training.
- PPO/GRPO.
- Process rewards.
- Learned legibility rewards.
- Real LLM agents.

## Stage 2: Task Diversity and Corrupted Handoff Robustness

Stage 2 expands the benchmark beyond arithmetic so tandem handoff behavior can be studied across different verifiable reasoning formats. Task diversity matters because a handoff method that works only for arithmetic may be exploiting a narrow answer format rather than producing generally useful reasoning.

Stage 2 also adds corrupted handoff evaluation. The senior still produces reasoning, but the trace is perturbed before it reaches the junior. This tests whether junior performance depends on a brittle exact trace or whether the handoff remains useful under mild noise.

New task families:

- List transformation tasks: sorting, filtering evens/odds, simple mapping, and reversing.
- Logic tasks: syllogisms, transitive comparison, and implication checks.
- Code tracing tasks: tiny generated assignment and loop-sum templates with known answers.

New evaluation mode:

- `corrupted_handoff`: senior reasoning is perturbed before the junior receives it.

New metrics:

- `corrupted_handoff_accuracy`
- `robustness_drop = tandem_handoff_accuracy - corrupted_handoff_accuracy`
- `handoff_gain = tandem_handoff_accuracy - junior_only_accuracy`
- accuracy by task type
- accuracy by difficulty
- failure counts by mode
- task counts by task type

Run the Stage 2 baseline:

```bash
python -m tandem_rlvr.experiments.run_stage2_baseline --num-tasks 300 --seed 42
```

Outputs:

```text
outputs/stage2_results.csv
outputs/stage2_summary.json
```

## Installation

```bash
cd tandem-rlvr
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Alternatively:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run the Stage 1 Baseline

```bash
python -m tandem_rlvr.experiments.run_stage1_baseline
```

The experiment prints a summary table and writes detailed per-task results to:

```text
outputs/results.csv
```

Example CLI options:

```bash
python -m tandem_rlvr.experiments.run_stage1_baseline --num-tasks 100 --difficulty mixed --seed 7
python -m tandem_rlvr.experiments.run_stage1_baseline --task-types addition multiplication
```

## Run Tests

```bash
pytest
```

## Project Structure

```text
tandem-rlvr/
  README.md
  pyproject.toml
  requirements.txt
  src/tandem_rlvr/
    tasks/
    agents/
    eval/
    experiments/
    utils/
  tests/
  outputs/
```

## Design Notes

The current agents are deliberately simple. `OracleSeniorAgent` produces compact reasoning across all generated task families, while `WeakJuniorAgent` has a limited direct-solving range and can use simple senior traces that expose a final result in natural language. This creates measurable gaps between junior-only, clean tandem handoff, and corrupted handoff performance without pretending to solve the full research problem.

Future stages:

- Stage 3: add LLM wrappers for local Hugging Face and API-based models.
- Stage 4: add explicit legibility and process-reward metrics.
- Stage 5: add an RLVR training loop.
- Stage 6: add generalization splits and ablations.
- Stage 7: write a paper-style report.
