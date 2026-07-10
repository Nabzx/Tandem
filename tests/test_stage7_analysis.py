import json
from pathlib import Path

import pandas as pd

from tandem_rlvr.analysis.load_results import LoadedResults, load_available_results, load_stage5_results, load_stage6_results
from tandem_rlvr.analysis.plots import generate_all_figures, generate_stage5_figures, generate_stage6_figures
from tandem_rlvr.analysis.report import generate_reports
from tandem_rlvr.experiments.run_stage7_generate_report import run_stage7_generate_report


def test_load_stage5_summary_and_results(tmp_path: Path) -> None:
    _write_json(tmp_path / "stage5_generalization_summary.json", _stage5_summary())
    pd.DataFrame(
        [
            {"split": "id_eval", "mode": "junior_only", "correct": False},
            {"split": "id_eval", "mode": "tandem_handoff", "correct": True},
        ]
    ).to_csv(tmp_path / "stage5_generalization_results.csv", index=False)
    _write_json(tmp_path / "stage5_process_summary.json", {"breakdowns": {"split": {}}})

    results = load_stage5_results(tmp_path)

    assert results.stage5_summary is not None
    assert results.stage5_results is not None
    assert results.stage5_process_summary is not None
    assert results.has_stage5()


def test_load_stage6_summary_and_results(tmp_path: Path) -> None:
    _write_json(tmp_path / "stage6_bandit_summary.json", _stage6_summary())
    _stage6_episodes().to_csv(tmp_path / "stage6_bandit_episodes.csv", index=False)
    _stage6_strategy_eval().to_csv(tmp_path / "stage6_strategy_eval.csv", index=False)
    _write_json(tmp_path / "stage6_strategy_eval_summary.json", {"mean_reward_by_strategy": {"structured_steps": 1.2}})

    results = load_stage6_results(tmp_path)

    assert results.stage6_summary is not None
    assert results.stage6_episodes is not None
    assert results.stage6_strategy_eval is not None
    assert results.stage6_strategy_summary is not None
    assert results.has_stage6()


def test_plot_functions_create_image_files_from_tiny_data(tmp_path: Path) -> None:
    results = LoadedResults(
        outputs_dir=tmp_path,
        stage5_summary=_stage5_summary(),
        stage5_process_summary={"breakdowns": {"split": {}}},
        stage6_episodes=_stage6_episodes(),
        stage6_strategy_eval=_stage6_strategy_eval(),
        stage6_strategy_summary={"mean_reward_by_strategy": {"structured_steps": 1.2}},
    )

    stage5_figures = generate_stage5_figures(results, tmp_path / "figures")
    stage6_figures = generate_stage6_figures(results, tmp_path / "figures")

    assert "stage5_accuracy_by_split" in stage5_figures
    assert "stage5_handoff_gain_by_split" in stage5_figures
    assert "stage6_strategy_reward" in stage6_figures
    assert "stage6_episode_rewards" in stage6_figures
    for path in [*stage5_figures.values(), *stage6_figures.values()]:
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG")


def test_report_generation_from_minimal_summaries(tmp_path: Path) -> None:
    results = LoadedResults(
        outputs_dir=tmp_path,
        stage5_summary=_stage5_summary(),
        stage6_summary=_stage6_summary(),
    )
    figures = {"stage5_accuracy_by_split": tmp_path / "figures" / "stage5_accuracy_by_split.png"}

    reports = generate_reports(results, tmp_path / "reports", figures)

    paper = reports["paper_report"].read_text(encoding="utf-8")
    project_summary = reports["project_summary"].read_text(encoding="utf-8")
    assert "## Abstract" in paper
    assert "RLVR-style Handoff Policy Optimization" in paper
    assert "Stage 6 episode-best strategy" in project_summary


def test_missing_output_files_are_handled_gracefully(tmp_path: Path) -> None:
    results = load_available_results(tmp_path)
    figures = generate_all_figures(results, tmp_path / "reports" / "figures")
    reports = generate_reports(results, tmp_path / "reports", figures)

    assert results.missing_files
    assert figures == {}
    assert reports["paper_report"].exists()
    assert "Missing optional output files" in reports["paper_report"].read_text(encoding="utf-8")


def test_stage7_script_smoke_test(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _write_json(outputs / "stage5_generalization_summary.json", _stage5_summary())
    pd.DataFrame(
        [
            {"split": "id_eval", "mode": "junior_only", "correct": False},
            {"split": "id_eval", "mode": "tandem_handoff", "correct": True},
        ]
    ).to_csv(outputs / "stage5_generalization_results.csv", index=False)
    _write_json(outputs / "stage5_process_summary.json", {"breakdowns": {"split": {}}})
    _write_json(outputs / "stage6_bandit_summary.json", _stage6_summary())
    _stage6_episodes().to_csv(outputs / "stage6_bandit_episodes.csv", index=False)
    _stage6_strategy_eval().to_csv(outputs / "stage6_strategy_eval.csv", index=False)
    _write_json(outputs / "stage6_strategy_eval_summary.json", {"mean_reward_by_strategy": {"structured_steps": 1.2}})

    result = run_stage7_generate_report(outputs, tmp_path / "reports")

    assert result["reports"]["paper_report"].exists()
    assert result["reports"]["project_summary"].exists()
    assert result["figures"]


def _stage5_summary() -> dict[str, object]:
    return {
        "accuracy_by_split_and_mode": {
            "id_eval": {"junior_only": 0.4, "tandem_handoff": 0.7, "senior_only": 0.9, "corrupted_handoff": 0.5},
            "ood_eval": {"junior_only": 0.3, "tandem_handoff": 0.5, "senior_only": 0.8, "corrupted_handoff": 0.2},
        },
        "handoff_gain_by_split": {"id_eval": 0.3, "ood_eval": 0.2},
        "process_reward_by_split": {"id_eval": 0.8, "ood_eval": 0.6},
        "ood_generalization_gap": 0.2,
        "stress_generalization_gap": None,
    }


def _stage6_summary() -> dict[str, object]:
    return {
        "episode_best_strategy": "worked_prefix",
        "episode_best_strategy_mean_reward": 1.0,
        "heldout_best_strategy": "structured_steps",
        "heldout_best_strategy_mean_reward": 1.2,
        "heldout_best_strategy_accuracy": 1.0,
        "default_strategy_comparison": {
            "heldout_best_vs_default_accuracy_delta": 0.0,
            "heldout_best_vs_default_reward_delta": 0.2,
            "heldout_best_vs_default_process_reward_delta": 0.1,
        },
        "warnings": ["Warning: this is a very small optimization run. Strategy rankings may be noisy."],
    }


def _stage6_episodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": 1,
                "strategy": "worked_prefix",
                "correct": True,
                "total_reward": 1.0,
                "process_reward_score": 0.8,
                "leaks_exact_answer": False,
                "hallucination_flag": False,
            },
            {
                "episode": 2,
                "strategy": "structured_steps",
                "correct": True,
                "total_reward": 1.2,
                "process_reward_score": 0.9,
                "leaks_exact_answer": False,
                "hallucination_flag": False,
            },
        ]
    )


def _stage6_strategy_eval() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "worked_prefix",
                "correct": True,
                "total_reward": 1.0,
                "process_reward_score": 0.8,
                "leaks_exact_answer": False,
                "hallucination_flag": False,
            },
            {
                "strategy": "structured_steps",
                "correct": True,
                "total_reward": 1.2,
                "process_reward_score": 0.9,
                "leaks_exact_answer": False,
                "hallucination_flag": False,
            },
        ]
    )


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")
