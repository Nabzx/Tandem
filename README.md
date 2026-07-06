# TandemRLVR

TandemRLVR is a small research infrastructure project for studying **Tandem Reinforcement Learning with Verifiable Rewards**: evaluating whether a stronger "senior" reasoning agent can produce intermediate reasoning that remains useful to a weaker "junior" agent.

This repository currently implements **Stage 1 only**: a minimal, testable evaluation scaffold for synthetic arithmetic tasks. It does not implement RL training, PPO/GRPO, learned rewards, or real language-model agents yet.

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
- Non-arithmetic task domains.

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

The current agents are deliberately simple. `OracleSeniorAgent` produces compact arithmetic reasoning, while `WeakJuniorAgent` has a limited direct-solving range and can use simple senior traces that expose a final numeric result in natural language. This creates a minimal measurable gap between junior-only and tandem handoff performance without pretending to solve the full research problem.

Future stages should replace these placeholders with real model wrappers while preserving the same interfaces.
