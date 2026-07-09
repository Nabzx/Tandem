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

For the fastest single-call check:

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 1 \
  --seed 42 \
  --senior-model llama3.2:1b \
  --junior-model llama3.2:1b \
  --quick \
  --modes senior_only \
  --verbose
```

Useful debugging flags:

- `--verbose` prints prompt previews, raw model output previews, parse status, normalized answers, and correctness.
- `--modes senior_only,junior_only,tandem,corrupted` lets you run only selected evaluation modes.
- `--max-generation-seconds 60` controls the Ollama client timeout used for each local generation.
- `--quick` uses short generations for local smoke tests.

Troubleshooting:

Local Ollama evaluation may look stuck if progress logging is disabled or if a model is slow, because each task can require multiple local model calls. Stage 3 now prints the task index, task type, and active mode as it runs. Use `--modes senior_only --num-tasks 1 --quick --verbose` to confirm that the model, parsing, and verification path work before launching a larger run.

Recommended debugging ladder:

Arithmetic senior-only sanity check:

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 10 \
  --seed 42 \
  --senior-model llama3.2:1b \
  --junior-model llama3.2:1b \
  --task-types arithmetic \
  --modes senior_only \
  --easy-only \
  --quick \
  --num-predict 192 \
  --verbose
```

Stronger senior comparison:

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 20 \
  --seed 42 \
  --senior-model llama3.1:latest \
  --junior-model llama3.2:1b \
  --task-types arithmetic \
  --modes senior_only,junior_only,tandem_handoff \
  --easy-only \
  --quick \
  --num-predict 192 \
  --verbose
```

`llama3.2:1b` is a tiny local model. Treat it as a weak junior or smoke-test model, not as a reliable senior reasoner.

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 5 \
  --seed 42 \
  --senior-model llama3.2:1b \
  --junior-model llama3.2:1b \
  --task-types arithmetic \
  --modes senior_only \
  --easy-only \
  --quick \
  --num-predict 192 \
  --verbose \
  --debug-save-prompts
```

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 5 \
  --seed 42 \
  --senior-model llama3.2:1b \
  --junior-model llama3.2:1b \
  --task-types arithmetic \
  --modes junior_only \
  --easy-only \
  --quick \
  --num-predict 192 \
  --verbose
```

```bash
python -m tandem_rlvr.experiments.run_stage3_llm_eval \
  --num-tasks 5 \
  --seed 42 \
  --senior-model llama3.2:1b \
  --junior-model llama3.2:1b \
  --task-types arithmetic \
  --modes tandem_handoff \
  --easy-only \
  --quick \
  --num-predict 192 \
  --verbose
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
- `senior_handoff` is an internal generation step, not a scored mode. In tandem evaluation, only the junior completion row is scored; the senior handoff is stored as metadata on `tandem_handoff` or `corrupted_handoff` rows.
- Modes that were not run are reported as `null` in `stage3_llm_summary.json`, not `0.0`. A null accuracy means skipped, not failed.
- Easy arithmetic is mainly a smoke test for local model, parser, and verifier wiring. It is usually too easy to measure meaningful handoff gain because senior-only, junior-only, and tandem can all be high.

If Ollama is unavailable, the experiment exits with a clear message asking you to install Ollama, run `ollama serve`, and pull a model such as `llama3.2:1b`.

Stage 3 does not perform RL fine-tuning yet. It evaluates real local LLMs inside the TandemRLVR scaffold. RLVR training will be introduced in a later stage.

## Stage 4: Legibility and Process-Reward Metrics

Final-answer accuracy is not enough for TandemRLVR. A senior model can help a junior by giving a clean, inspectable handoff, or it can appear helpful by leaking the final answer, adding irrelevant assumptions, or producing brittle reasoning that fails under corruption. Stage 4 adds deterministic, local process metrics for evaluating the senior reasoning trace itself.

Stage 4 metrics include:

- Legibility metrics: word count, sentence count, average sentence length, too-short/too-long flags, operation hints, step markers, and a heuristic `legibility_score`.
- Leakage metrics: whether the handoff reveals the expected answer exactly or in normalized form, with special handling for integers, lists, booleans, and text answers.
- Relevance and hallucination metrics: whether reasoning mentions task entities and task operations, plus flags for suspicious phrases such as `go over 100`, `divide by 1`, `external data`, or `not enough information`.
- Usefulness metrics: whether tandem handoff improves, hurts, or remains unchanged relative to junior-only performance, and whether corrupted handoff hurts relative to clean tandem handoff.
- Process reward score: a weighted heuristic combination:

```text
0.30 * legibility_score
+ 0.25 * leakage_score
+ 0.25 * relevance_score
+ 0.20 * usefulness_score
```

If usefulness is unavailable for a row, the score is computed from available components and the available component names are recorded. These metrics are heuristic diagnostics, not ground truth. They are intended as candidate reward signals for later RLVR experiments.

Run Stage 4 on Stage 3 results:

```bash
python -m tandem_rlvr.experiments.run_stage4_process_metrics \
  --input outputs/stage3_llm_results.csv \
  --output-dir outputs
```

Outputs:

```text
outputs/stage4_process_metrics.csv
outputs/stage4_process_summary.json
```

## Stage 5: Generalization and OOD Evaluation

Stage 5 tests whether tandem handoff helps only on easy in-distribution examples or continues to help under distribution shift. This matters because weak-overseer handoff should be robust to harder tasks, edge cases, and shifted formats rather than only improving familiar tasks.

Splits:

- `train`: reserved for future training-style use, currently generated but not trained on.
- `id_eval`: in-distribution evaluation with smaller, simpler tasks.
- `ood_eval`: shifted evaluation with larger numbers, negatives, duplicates, false logic cases, variable overwrites, and other controlled shifts.
- `stress_eval`: harder synthetic edge cases such as distractor wording, longer chains, repeated values, zero/one arithmetic, empty-list-like cases, and longer code traces.

Each Stage 5 task records:

```text
split
distribution
ood_type
task_family
```

Stage 5 measures accuracy, handoff gain, robustness drop, and process-reward metrics by split. `ood_generalization_gap` is the difference between `id_eval` tandem accuracy and `ood_eval` tandem accuracy. `stress_generalization_gap` compares `id_eval` tandem accuracy to `stress_eval` tandem accuracy. Missing splits or modes are reported as `null`, not zero.

Run Stage 5:

```bash
python -m tandem_rlvr.experiments.run_stage5_generalization_eval \
  --num-tasks-per-split 30 \
  --seed 42 \
  --senior-model llama3.1:latest \
  --junior-model llama3.2:1b \
  --splits id_eval,ood_eval,stress_eval \
  --modes senior_only,junior_only,tandem_handoff,corrupted_handoff \
  --quick \
  --num-predict 192
```

Outputs:

```text
outputs/stage5_generalization_results.csv
outputs/stage5_generalization_summary.json
outputs/stage5_process_metrics.csv
outputs/stage5_process_summary.json
```

Stage 5 still does not perform RL training. It evaluates generalization behavior before introducing RLVR.

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
    metrics/
    experiments/
    utils/
  tests/
  outputs/
```

## Design Notes

The current agents are deliberately simple. `OracleSeniorAgent` produces compact reasoning across all generated task families, while `WeakJuniorAgent` has a limited direct-solving range and can use simple senior traces that expose a final result in natural language. This creates measurable gaps between junior-only, clean tandem handoff, and corrupted handoff performance without pretending to solve the full research problem.

Future stages:

- Stage 6: add an RLVR training loop.
- Stage 7: compare standard RLVR vs TandemRLVR.
- Stage 8: write a paper-style report with plots, ablations, and limitations.
