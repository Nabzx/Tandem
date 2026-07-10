# TandemRLVR Project Summary

TandemRLVR is a research-quality Python project studying whether stronger reasoning agents can produce intermediate traces that remain useful and legible to weaker overseers. The project builds synthetic verifiable tasks, local Ollama-based senior/junior agents, tandem handoff evaluation, process metrics, OOD generalization tests, and a lightweight RLVR-style bandit over handoff strategies.

## What Was Built

- Verifiable arithmetic, list, logic, and code-tracing task environments.
- Senior-only, junior-only, tandem handoff, and corrupted handoff evaluations.
- Diagnostic metrics for legibility, leakage, relevance, hallucination, and usefulness.
- ID/OOD/stress split evaluation.
- Stage 6 handoff strategy optimization using contextual bandits.
- Stage 7 report and figure generation from saved outputs.

## Why It Matters

The project targets a core scalable-oversight question: can stronger models help weaker overseers without hiding the work or simply leaking the answer? This is relevant to RL and RLVR because the process metrics can act as transparent reward signals for optimizing helpful, inspectable intermediate reasoning.

## Headline Results

Stage 5 handoff gains by split were id_eval=0.100, ood_eval=0.100, stress_eval=-0.100. Stage 6 episode-best strategy was `structured_steps`, while heldout-best strategy was `structured_steps`.

## Reproduce

```bash
pytest
python -m tandem_rlvr.experiments.run_stage7_generate_report --outputs-dir outputs --report-dir reports
```

## Next Steps

Run larger multi-seed Stage 6 experiments, compare against full RLVR fine-tuning baselines, and strengthen process-reward validation against reward hacking and answer leakage.
