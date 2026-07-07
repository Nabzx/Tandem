# TandemRLVR

TandemRLVR is a small research infrastructure project for studying **Tandem Reinforcement Learning with Verifiable Rewards**: evaluating whether a stronger "senior" reasoning agent can produce intermediate reasoning that remains useful to a weaker "junior" agent.

This repository currently implements **Stages 1-3**: a minimal, testable evaluation scaffold with synthetic tasks, tandem handoff evaluation, corrupted-handoff robustness tests, and local LLM agent wrappers through Ollama. It does not implement RL training, PPO/GRPO, or learned rewards yet.

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

- Hugging Face or API model integrations.
- RL training.
- PPO/GRPO.
- Process rewards.
- Learned legibility rewards.

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

## Stage 3: Local LLM Agent Evaluation

Stage 3 moves beyond heuristic agents by adding local LLM wrappers. This makes it possible to ask the core TandemRLVR question with real model behavior: can a stronger local model produce reasoning traces that help a weaker local model solve verifiable tasks?

Ollama is used first because it is free, local, lightweight to integrate, and supports configurable model names such as `llama3.2:1b`, `llama3.2:3b`, `llama3.1:8b`, `mistral`, `gemma3`, and `qwen2.5`. The heuristic agents remain available as deterministic baselines.

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama from <https://ollama.com>, start the local server, then pull models:

```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
```

Run a local LLM evaluation:

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 50 \
  --seed 42 \
  --senior-model llama3.2:3b \
  --junior-model llama3.2:1b
```

Quick smoke run:

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 20 \
  --seed 42 \
  --senior-model llama3.2:1b \
  --junior-model llama3.2:1b \
  --quick
```

Outputs:

```text
outputs/stage3_llm_results.csv
outputs/stage3_llm_summary.json
```

Interpretation:

- `handoff_gain` measures how much tandem handoff improves over junior-only performance.
- `robustness_drop` measures how much performance falls when the senior reasoning is perturbed.
- Parse-status counts help separate model reasoning failures from output-format failures.

If Ollama is unavailable, the experiment exits with a clear message asking you to install Ollama, run `ollama serve`, and pull a model such as `llama3.2:1b`.

Stage 3 does not perform RL fine-tuning yet. It evaluates real local LLMs inside the TandemRLVR scaffold. RLVR training will be introduced in a later stage.

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

- Stage 4: add legibility and process-reward metrics.
- Stage 5: add generalization splits and out-of-distribution evaluations.
- Stage 6: add an RLVR training loop.
- Stage 7: compare standard RLVR vs TandemRLVR.
- Stage 8: write a paper-style report with plots, ablations, and limitations.
