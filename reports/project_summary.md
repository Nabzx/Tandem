# TandemRLVR Project Summary

TandemRLVR is a local, reproducible research framework for studying whether stronger reasoning models can produce intermediate traces that remain useful to weaker overseers. It uses verifiable synthetic tasks, local Ollama senior/junior agents, tandem handoff evaluation, process-reward diagnostics, ID/OOD/stress generalization splits, and a lightweight RLVR-style contextual bandit over handoff strategies. The goal is not to claim frontier-model training, but to build a concrete empirical scaffold for weak-overseer compatibility.

## Technical Contribution

The project separates final-answer correctness from handoff quality. It evaluates whether a junior model improves when given senior reasoning, whether that improvement survives corrupted handoffs and distribution shift, and whether transparent process metrics can serve as candidate reward signals. Stage 6 optimizes prompt-strategy selection with a bandit; it does not fine-tune LLM weights.

## Headline Results

Stage 5 handoff gains by split were id_eval=0.100, ood_eval=0.100, stress_eval=-0.100. Stage 6 episode-best strategy was `structured_steps`, while heldout-best strategy was `structured_steps`. In the small Stage 5 run, tandem handoff improved ID and OOD accuracy but hurt stress-split accuracy, suggesting that handoff can help while remaining brittle under harder shifts. Stage 6 smoke runs verify that the reward-and-bandit loop executes, but strategy rankings should be treated as noisy until larger multi-seed runs are completed.

## Limitations

Experiments use small local models and synthetic tasks. Process rewards are heuristic rather than learned or human-validated. Sample sizes are limited, and prompt-policy optimization is not equivalent to training model weights.

## Reproduce

```bash
pytest
python -m tandem_rlvr.experiments.run_stage7_generate_report --outputs-dir outputs --report-dir reports
```

## Next Steps

Run larger multi-seed Stage 6 experiments, learn or calibrate process rewards, add a full RLVR/PPO/GRPO fine-tuning baseline, expand to harder code and research-agent tasks, and stress-test leakage and reward hacking.
