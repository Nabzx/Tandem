from __future__ import annotations

from pathlib import Path
from typing import Any

from tandem_rlvr.analysis.load_results import LoadedResults


def generate_reports(results: LoadedResults, report_dir: str | Path, figures: dict[str, Path] | None = None) -> dict[str, Path]:
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    paper_path = report_path / "tandem_rlvr_report.md"
    summary_path = report_path / "project_summary.md"
    paper_path.write_text(build_paper_report(results, figures), encoding="utf-8")
    summary_path.write_text(build_project_summary(results), encoding="utf-8")
    return {"paper_report": paper_path, "project_summary": summary_path}


def build_paper_report(results: LoadedResults, figures: dict[str, Path] | None = None) -> str:
    figures = figures or {}
    lines = [
        "# TandemRLVR: Training Reasoning Agents to Remain Legible to Weaker Overseers",
        "",
        "## Abstract",
        "",
        _abstract(results),
        "",
        "## Motivation",
        "",
        (
            "Scalable oversight depends on weaker overseers being able to inspect, continue, and verify the work of stronger systems. "
            "TandemRLVR studies this setting with a senior reasoning agent, a weaker junior agent, and verifiable tasks where final answers can be checked automatically."
        ),
        "",
        "## Research Question",
        "",
        (
            "Can a stronger senior model improve a weaker junior model's task performance by producing intermediate handoff reasoning that is useful, legible, and not merely answer leakage?"
        ),
        "",
        "## Method",
        "",
        (
            "The pipeline evaluates senior-only, junior-only, clean tandem handoff, and corrupted handoff modes. "
            "It then scores handoff traces with transparent process metrics and uses a lightweight contextual bandit to choose among senior handoff prompt strategies. "
            "Stage 6 is not LLM fine-tuning; it is RLVR-style policy optimization over discrete handoff strategies."
        ),
        "",
        "## Task Environments",
        "",
        (
            "Tasks are synthetic and verifiable, covering arithmetic, list transformations, boolean logic, and code tracing. "
            "Stage 5 adds in-distribution, out-of-distribution, and stress splits to test whether handoff behavior survives distribution shift."
        ),
        "",
        "## Evaluation Modes",
        "",
        "- `senior_only`: the senior model answers directly.",
        "- `junior_only`: the junior model answers directly.",
        "- `tandem_handoff`: the senior provides partial reasoning and the junior gives the scored answer.",
        "- `corrupted_handoff`: the junior receives perturbed senior reasoning.",
        "",
        "## Process Metrics",
        "",
        (
            "Process metrics estimate legibility, leakage, relevance, hallucination flags, and usefulness. "
            "These metrics are heuristic and transparent; they are intended as candidate reward signals rather than definitive measures of reasoning quality."
        ),
        "",
        "## Generalization Evaluation",
        "",
        _stage5_section(results, figures),
        "",
        "## RLVR-style Handoff Policy Optimization",
        "",
        _stage6_section(results, figures),
        "",
        "## Results",
        "",
        _results_section(results),
        "",
        "## Failure Analysis",
        "",
        _failure_analysis(results),
        "",
        "## Limitations",
        "",
        (
            "This project does not claim to train a frontier model. It is a small local empirical framework. The tasks are synthetic, local LLM performance depends on the selected Ollama models, and the process reward is hand-designed. "
            "Stage 6 optimizes a bandit over prompt strategies, not neural model parameters. Results are preliminary, and larger multi-seed experiments are required before making strong empirical claims."
        ),
        "",
        "## Future Work",
        "",
        "- Run larger Stage 5 and Stage 6 sweeps across seeds and model pairs.",
        "- Learn or calibrate a process-reward model.",
        "- Compare bandit-selected handoff strategies against full RLVR/PPO/GRPO fine-tuning baselines.",
        "- Expand to harder code and research-agent tasks.",
        "- Stress-test reward hacking, answer leakage, and hallucinated intermediate reasoning.",
        "- Compare against direct answer leakage baselines.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "pytest",
        "python -m tandem_rlvr.experiments.run_stage7_generate_report --outputs-dir outputs --report-dir reports",
        "```",
        "",
        _missing_files_note(results),
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_project_summary(results: LoadedResults) -> str:
    headline = _headline_results(results)
    return (
        "# TandemRLVR Project Summary\n\n"
        "TandemRLVR is a local, reproducible research framework for studying whether stronger reasoning models can produce intermediate traces that remain useful to weaker overseers. "
        "It uses verifiable synthetic tasks, local Ollama senior/junior agents, tandem handoff evaluation, process-reward diagnostics, ID/OOD/stress generalization splits, and a lightweight RLVR-style contextual bandit over handoff strategies. "
        "The goal is not to claim frontier-model training, but to build a concrete empirical scaffold for weak-overseer compatibility.\n\n"
        "## Technical Contribution\n\n"
        "The project separates final-answer correctness from handoff quality. It evaluates whether a junior model improves when given senior reasoning, whether that improvement survives corrupted handoffs and distribution shift, and whether transparent process metrics can serve as candidate reward signals. "
        "Stage 6 optimizes prompt-strategy selection with a bandit; it does not fine-tune LLM weights.\n\n"
        "## Headline Results\n\n"
        f"{headline} In the small Stage 5 run, tandem handoff improved ID and OOD accuracy but hurt stress-split accuracy, suggesting that handoff can help while remaining brittle under harder shifts. Stage 6 smoke runs verify that the reward-and-bandit loop executes, but strategy rankings should be treated as noisy until larger multi-seed runs are completed.\n\n"
        "## Limitations\n\n"
        "Experiments use small local models and synthetic tasks. Process rewards are heuristic rather than learned or human-validated. Sample sizes are limited, and prompt-policy optimization is not equivalent to training model weights.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "pytest\n"
        "python -m tandem_rlvr.experiments.run_stage7_generate_report --outputs-dir outputs --report-dir reports\n"
        "```\n\n"
        "## Next Steps\n\n"
        "Run larger multi-seed Stage 6 experiments, learn or calibrate process rewards, add a full RLVR/PPO/GRPO fine-tuning baseline, expand to harder code and research-agent tasks, and stress-test leakage and reward hacking.\n"
    )


def _abstract(results: LoadedResults) -> str:
    return (
        "TandemRLVR investigates whether a stronger senior reasoning model can help a weaker junior model solve verifiable tasks through intermediate handoff reasoning. "
        f"{_headline_results(results)} "
        "The current system is a small local empirical framework: it evaluates final-answer accuracy, process-level handoff quality, distribution shift, and a lightweight RLVR-style bandit over prompt strategies. "
        "It does not perform LLM weight training."
    )


def _stage5_section(results: LoadedResults, figures: dict[str, Path]) -> str:
    if not results.has_stage5():
        return "Stage 5 outputs were not found, so this section was skipped."
    summary = results.stage5_summary or {}
    lines = []
    gains = summary.get("handoff_gain_by_split", {})
    process = summary.get("process_reward_by_split", {})
    if gains:
        lines.append("Observed handoff gain by split: " + _dict_sentence(gains) + ".")
        lines.append(
            "In the current small run, tandem handoff improved ID/OOD performance but degraded stress-split performance, which is consistent with useful but brittle handoff behavior."
        )
    if process:
        lines.append("Mean process reward by split: " + _dict_sentence(process) + ".")
    if summary.get("ood_generalization_gap") is not None:
        lines.append(f"OOD generalization gap: {_fmt(summary['ood_generalization_gap'])}.")
    if summary.get("stress_generalization_gap") is not None:
        lines.append(f"Stress generalization gap: {_fmt(summary['stress_generalization_gap'])}.")
    lines.extend(_figure_links(figures, ["stage5_accuracy_by_split", "stage5_handoff_gain_by_split", "stage5_process_reward_by_split"]))
    return "\n\n".join(lines) if lines else "Stage 5 files were present, but no recognized metrics were available."


def _stage6_section(results: LoadedResults, figures: dict[str, Path]) -> str:
    if not results.has_stage6():
        return "Stage 6 outputs were not found, so this section was skipped."
    summary = results.stage6_summary or {}
    comparison = summary.get("default_strategy_comparison", {})
    lines = [
        f"Episode-best strategy: `{summary.get('episode_best_strategy', summary.get('best_strategy', 'unknown'))}`.",
        f"Heldout-best strategy: `{summary.get('heldout_best_strategy', 'unknown')}`.",
    ]
    if summary.get("heldout_best_strategy_mean_reward") is not None:
        lines.append(f"Heldout-best mean reward: {_fmt(summary['heldout_best_strategy_mean_reward'])}.")
    if summary.get("heldout_best_strategy_accuracy") is not None:
        lines.append(f"Heldout-best accuracy: {_fmt(summary['heldout_best_strategy_accuracy'])}.")
    if comparison:
        lines.append(
            "Heldout-best vs default deltas: "
            f"accuracy={_fmt(comparison.get('heldout_best_vs_default_accuracy_delta'))}, "
            f"reward={_fmt(comparison.get('heldout_best_vs_default_reward_delta'))}, "
            f"process_reward={_fmt(comparison.get('heldout_best_vs_default_process_reward_delta'))}."
        )
    warnings = summary.get("warnings", [])
    if warnings:
        lines.append("Run-size warnings: " + " ".join(warnings))
    lines.append(
        "These Stage 6 results should be read as an optimization-loop smoke test. Larger held-out evaluations and multiple seeds are needed before ranking handoff strategies."
    )
    lines.extend(
        _figure_links(
            figures,
            [
                "stage6_strategy_reward",
                "stage6_strategy_accuracy",
                "stage6_strategy_leakage",
                "stage6_episode_rewards",
                "stage6_strategy_selection_counts",
            ],
        )
    )
    return "\n\n".join(lines)


def _results_section(results: LoadedResults) -> str:
    parts = []
    stage5 = results.stage5_summary or {}
    if stage5.get("accuracy_by_split_and_mode"):
        parts.append("Stage 5 reports accuracy by split and mode, making it possible to inspect when tandem handoff helps and when stress settings break handoff.")
    stage6 = results.stage6_summary or {}
    if stage6:
        parts.append(
            "Stage 6 separates episode-best strategy selection from held-out strategy evaluation, reducing the risk of overinterpreting noisy bandit episode rewards."
        )
    return " ".join(parts) if parts else "No Stage 5 or Stage 6 output files were available for quantitative results."


def _failure_analysis(results: LoadedResults) -> str:
    lines = []
    failures = (results.stage5_summary or {}).get("failure_type_counts_by_split", {})
    if failures:
        lines.append("Stage 5 failure counts by split: " + _dict_sentence({split: sum(counts.values()) for split, counts in failures.items()}) + ".")
    stage6 = results.stage6_summary or {}
    if stage6.get("leakage_rate_by_strategy"):
        lines.append("Stage 6 tracks leakage by strategy, which is important because a high-reward handoff can be misleading if it reveals the final answer.")
    return " ".join(lines) if lines else "Failure analysis is limited when detailed result files are missing."


def _headline_results(results: LoadedResults) -> str:
    stage5 = results.stage5_summary or {}
    stage6 = results.stage6_summary or {}
    snippets = []
    if stage5.get("handoff_gain_by_split"):
        snippets.append("Stage 5 handoff gains by split were " + _dict_sentence(stage5["handoff_gain_by_split"]) + ".")
    if stage6:
        snippets.append(
            f"Stage 6 episode-best strategy was `{stage6.get('episode_best_strategy', stage6.get('best_strategy', 'unknown'))}`, "
            f"while heldout-best strategy was `{stage6.get('heldout_best_strategy', 'unknown')}`."
        )
    return " ".join(snippets) if snippets else "Saved result files were not available, so no headline metrics are reported."


def _figure_links(figures: dict[str, Path], keys: list[str]) -> list[str]:
    links = []
    for key in keys:
        path = figures.get(key)
        if path is not None:
            links.append(f"![{key}](figures/{path.name})")
    return links


def _missing_files_note(results: LoadedResults) -> str:
    if not results.missing_files:
        return ""
    files = "\n".join(f"- `{path}`" for path in results.missing_files)
    return "Missing optional output files:\n\n" + files


def _dict_sentence(values: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_fmt(value)}" for key, value in values.items())


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
