from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pandas as pd

from tandem_rlvr.analysis.load_results import LoadedResults


FIGURE_FILENAMES = {
    "stage5_accuracy_by_split": "stage5_accuracy_by_split.png",
    "stage5_handoff_gain_by_split": "stage5_handoff_gain_by_split.png",
    "stage5_process_reward_by_split": "stage5_process_reward_by_split.png",
    "stage6_strategy_reward": "stage6_strategy_reward.png",
    "stage6_strategy_accuracy": "stage6_strategy_accuracy.png",
    "stage6_strategy_leakage": "stage6_strategy_leakage.png",
    "stage6_episode_rewards": "stage6_episode_rewards.png",
    "stage6_strategy_selection_counts": "stage6_strategy_selection_counts.png",
}


def generate_all_figures(results: LoadedResults, figures_dir: str | Path) -> dict[str, Path]:
    generated: dict[str, Path] = {}
    generated.update(generate_stage5_figures(results, figures_dir))
    generated.update(generate_stage6_figures(results, figures_dir))
    return generated


def generate_stage5_figures(results: LoadedResults, figures_dir: str | Path) -> dict[str, Path]:
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    accuracy = _stage5_accuracy_table(results.stage5_summary, results.stage5_results)
    if not accuracy.empty:
        path = figures_path / FIGURE_FILENAMES["stage5_accuracy_by_split"]
        _plot_grouped_bar(
            accuracy,
            path,
            title="Stage 5 Accuracy by Split and Mode",
            ylabel="Accuracy",
            ylim=(0, 1),
        )
        generated["stage5_accuracy_by_split"] = path

    handoff_gain = _stage5_handoff_gain_table(results.stage5_summary, accuracy)
    if not handoff_gain.empty:
        path = figures_path / FIGURE_FILENAMES["stage5_handoff_gain_by_split"]
        _plot_bar(
            handoff_gain,
            "split",
            "handoff_gain",
            path,
            title="Stage 5 Handoff Gain by Split",
            ylabel="Tandem - Junior Accuracy",
        )
        generated["stage5_handoff_gain_by_split"] = path

    process_reward = _stage5_process_reward_table(results.stage5_summary, results.stage5_process_summary)
    if not process_reward.empty:
        path = figures_path / FIGURE_FILENAMES["stage5_process_reward_by_split"]
        _plot_bar(
            process_reward,
            "split",
            "process_reward",
            path,
            title="Stage 5 Process Reward by Split",
            ylabel="Mean Process Reward",
            ylim=(0, 1),
        )
        generated["stage5_process_reward_by_split"] = path

    return generated


def generate_stage6_figures(results: LoadedResults, figures_dir: str | Path) -> dict[str, Path]:
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    strategy_eval = results.stage6_strategy_eval
    episodes = results.stage6_episodes

    strategy_metrics = _stage6_strategy_metrics(strategy_eval, results.stage6_strategy_summary)
    if not strategy_metrics.empty:
        for key, column, title, ylabel in (
            ("stage6_strategy_reward", "heldout_mean_reward", "Stage 6 Held-out Mean Reward by Strategy", "Mean Reward"),
            ("stage6_strategy_accuracy", "heldout_accuracy", "Stage 6 Held-out Accuracy by Strategy", "Accuracy"),
            ("stage6_strategy_leakage", "heldout_leakage_rate", "Stage 6 Held-out Leakage Rate by Strategy", "Leakage Rate"),
        ):
            if column in strategy_metrics.columns:
                path = figures_path / FIGURE_FILENAMES[key]
                _plot_bar(strategy_metrics, "strategy", column, path, title=title, ylabel=ylabel, ylim=(0, 1) if "accuracy" in column or "leakage" in column else None)
                generated[key] = path

    if episodes is not None and not episodes.empty and {"episode", "total_reward"}.issubset(episodes.columns):
        path = figures_path / FIGURE_FILENAMES["stage6_episode_rewards"]
        _plot_line(
            episodes.sort_values("episode"),
            path,
            title="Stage 6 Episode Reward Over Time",
            ylabel="Reward",
        )
        generated["stage6_episode_rewards"] = path

    if episodes is not None and not episodes.empty and "strategy" in episodes.columns:
        counts = episodes["strategy"].value_counts().sort_index().rename_axis("strategy").reset_index(name="selected_count")
        path = figures_path / FIGURE_FILENAMES["stage6_strategy_selection_counts"]
        _plot_bar(
            counts,
            "strategy",
            "selected_count",
            path,
            title="Stage 6 Strategy Selection Counts",
            ylabel="Selected Count",
        )
        generated["stage6_strategy_selection_counts"] = path

    return generated


def _stage5_accuracy_table(summary: dict[str, Any] | None, results: pd.DataFrame | None) -> pd.DataFrame:
    nested = (summary or {}).get("accuracy_by_split_and_mode", {})
    if nested:
        return pd.DataFrame.from_dict(nested, orient="index").rename_axis("split").reset_index()
    if results is None or results.empty or not {"split", "mode", "correct"}.issubset(results.columns):
        return pd.DataFrame()
    return results.pivot_table(index="split", columns="mode", values="correct", aggfunc="mean").reset_index()


def _stage5_handoff_gain_table(summary: dict[str, Any] | None, accuracy: pd.DataFrame) -> pd.DataFrame:
    gains = (summary or {}).get("handoff_gain_by_split", {})
    if gains:
        return pd.DataFrame({"split": list(gains), "handoff_gain": list(gains.values())})
    if accuracy.empty or not {"split", "junior_only", "tandem_handoff"}.issubset(accuracy.columns):
        return pd.DataFrame()
    return pd.DataFrame({"split": accuracy["split"], "handoff_gain": accuracy["tandem_handoff"] - accuracy["junior_only"]})


def _stage5_process_reward_table(summary: dict[str, Any] | None, process_summary: dict[str, Any] | None) -> pd.DataFrame:
    rewards = (summary or {}).get("process_reward_by_split", {})
    if rewards:
        return pd.DataFrame({"split": list(rewards), "process_reward": list(rewards.values())})
    split_breakdown = ((process_summary or {}).get("breakdowns", {}) or {}).get("split", {})
    if split_breakdown:
        return pd.DataFrame(
            {
                "split": list(split_breakdown),
                "process_reward": [values.get("mean_process_reward_score") for values in split_breakdown.values()],
            }
        )
    return pd.DataFrame()


def _stage6_strategy_metrics(strategy_eval: pd.DataFrame | None, strategy_summary: dict[str, Any] | None) -> pd.DataFrame:
    if strategy_eval is not None and not strategy_eval.empty and "strategy" in strategy_eval.columns:
        grouped = strategy_eval.groupby("strategy")
        return pd.DataFrame(
            {
                "strategy": list(grouped.groups),
                "heldout_mean_reward": grouped["total_reward"].mean().values if "total_reward" in strategy_eval.columns else None,
                "heldout_accuracy": grouped["correct"].mean().values if "correct" in strategy_eval.columns else None,
                "heldout_leakage_rate": grouped["leaks_exact_answer"].mean().values if "leaks_exact_answer" in strategy_eval.columns else None,
            }
        ).dropna(axis=1, how="all")

    summary = strategy_summary or {}
    rewards = summary.get("mean_reward_by_strategy", {})
    accuracies = summary.get("accuracy_by_strategy", {})
    if not rewards and not accuracies:
        return pd.DataFrame()
    strategies = sorted(set(rewards) | set(accuracies))
    return pd.DataFrame(
        {
            "strategy": strategies,
            "heldout_mean_reward": [rewards.get(strategy) for strategy in strategies],
            "heldout_accuracy": [accuracies.get(strategy) for strategy in strategies],
        }
    )


def _plot_grouped_bar(df: pd.DataFrame, path: Path, title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    columns = [column for column in df.columns if column != "split"]
    if not columns:
        _write_placeholder_png(path)
        return
    plt = _load_pyplot()
    if plt is None:
        _write_placeholder_png(path)
        return
    ax = df.set_index("split")[columns].plot(kind="bar", figsize=(8, 4.8), width=0.78)
    _finish_axes(plt, ax, title, ylabel, ylim)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    plt = _load_pyplot()
    if plt is None:
        _write_placeholder_png(path)
        return
    plot_df = df[[x, y]].dropna().copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(plot_df[x].astype(str), plot_df[y].astype(float), color="#4C78A8")
    _finish_axes(plt, ax, title, ylabel, ylim)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_line(df: pd.DataFrame, path: Path, title: str, ylabel: str) -> None:
    plt = _load_pyplot()
    if plt is None:
        _write_placeholder_png(path)
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(df["episode"], df["total_reward"], marker="o", linewidth=1.6, color="#4C78A8", label="episode reward")
    if len(df) >= 5:
        rolling = df["total_reward"].rolling(window=min(5, len(df)), min_periods=1).mean()
        ax.plot(df["episode"], rolling, linewidth=2.2, color="#F58518", label="rolling mean")
        ax.legend(frameon=False)
    _finish_axes(plt, ax, title, ylabel)
    ax.set_xlabel("Episode")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finish_axes(plt: Any, ax: Any, title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()


def _load_pyplot() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _write_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Valid 1x1 transparent PNG. Used only when matplotlib has not been installed yet.
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    path.write_bytes(data)
