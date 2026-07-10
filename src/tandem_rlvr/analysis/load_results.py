from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LoadedResults:
    outputs_dir: Path
    stage5_summary: dict[str, Any] | None = None
    stage5_results: pd.DataFrame | None = None
    stage5_process_summary: dict[str, Any] | None = None
    stage6_summary: dict[str, Any] | None = None
    stage6_episodes: pd.DataFrame | None = None
    stage6_strategy_eval: pd.DataFrame | None = None
    stage6_strategy_summary: dict[str, Any] | None = None
    missing_files: list[str] = field(default_factory=list)

    def has_stage5(self) -> bool:
        return self.stage5_summary is not None or self.stage5_results is not None or self.stage5_process_summary is not None

    def has_stage6(self) -> bool:
        return self.stage6_summary is not None or self.stage6_episodes is not None or self.stage6_strategy_eval is not None


def load_available_results(outputs_dir: str | Path) -> LoadedResults:
    outputs_path = Path(outputs_dir)
    results = LoadedResults(outputs_dir=outputs_path)
    _merge(results, load_stage5_results(outputs_path))
    _merge(results, load_stage6_results(outputs_path))
    return results


def load_stage5_results(outputs_dir: str | Path) -> LoadedResults:
    outputs_path = Path(outputs_dir)
    results = LoadedResults(outputs_dir=outputs_path)
    results.stage5_summary = _load_json(outputs_path / "stage5_generalization_summary.json", results)
    results.stage5_results = _load_csv(outputs_path / "stage5_generalization_results.csv", results)
    results.stage5_process_summary = _load_json(outputs_path / "stage5_process_summary.json", results)
    return results


def load_stage6_results(outputs_dir: str | Path) -> LoadedResults:
    outputs_path = Path(outputs_dir)
    results = LoadedResults(outputs_dir=outputs_path)
    results.stage6_summary = _load_json(outputs_path / "stage6_bandit_summary.json", results)
    results.stage6_episodes = _load_csv(outputs_path / "stage6_bandit_episodes.csv", results)
    results.stage6_strategy_eval = _load_csv(outputs_path / "stage6_strategy_eval.csv", results)
    results.stage6_strategy_summary = _load_json(outputs_path / "stage6_strategy_eval_summary.json", results)
    return results


def _load_json(path: Path, results: LoadedResults) -> dict[str, Any] | None:
    if not path.exists():
        results.missing_files.append(str(path))
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path, results: LoadedResults) -> pd.DataFrame | None:
    if not path.exists():
        results.missing_files.append(str(path))
        return None
    return pd.read_csv(path)


def _merge(target: LoadedResults, source: LoadedResults) -> None:
    for field_name in (
        "stage5_summary",
        "stage5_results",
        "stage5_process_summary",
        "stage6_summary",
        "stage6_episodes",
        "stage6_strategy_eval",
        "stage6_strategy_summary",
    ):
        value = getattr(source, field_name)
        if value is not None:
            setattr(target, field_name, value)
    target.missing_files.extend(source.missing_files)
