from pathlib import Path

from tandem_rlvr.experiments.run_stage2_baseline import build_mixed_benchmark, main


def test_build_mixed_benchmark_spans_task_families() -> None:
    tasks = build_mixed_benchmark(8, seed=42)

    prefixes = {task.task_id.split("-")[0] for task in tasks}

    assert {"arith", "list", "logic", "code"}.issubset(prefixes)


def test_stage2_experiment_smoke(monkeypatch, tmp_path: Path) -> None:
    results_path = tmp_path / "stage2_results.csv"
    summary_path = tmp_path / "stage2_summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_stage2_baseline",
            "--num-tasks",
            "12",
            "--seed",
            "42",
            "--results-path",
            str(results_path),
            "--summary-path",
            str(summary_path),
        ],
    )

    main()

    assert results_path.exists()
    assert summary_path.exists()
