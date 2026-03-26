import re
from pathlib import Path
from typing import Dict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..evaluation import validate_distances
from .discovery import (
    ExperimentEntry,
    collect_experiments,
    get_experiment_names,
    group_by_query,
)
from .paths import get_workspace_root


def _parse_algorithm_name(variant: str) -> str:
    match = re.match(r"(\w+)_", variant)
    return match.group(1) if match else variant


def _is_throughput_df(df: pd.DataFrame) -> bool:
    return "throughput" in df.columns


def _partition_variant_dfs(
    variant_dfs: Dict[str, pd.DataFrame],
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    throughput_dfs = {}
    time_dfs = {}
    for name, df in variant_dfs.items():
        if _is_throughput_df(df):
            throughput_dfs[name] = df
        else:
            time_dfs[name] = df
    return throughput_dfs, time_dfs


def _plot_histogram(variant_dfs, title, output_path, variant_filter=None):
    if variant_filter is not None:
        variant_dfs = {
            name: df for name, df in variant_dfs.items() if variant_filter in name
        }
    fig, ax = plt.subplots(figsize=(10, 6))

    all_times = np.concatenate([df["time"].values for df in variant_dfs.values()])
    p99 = np.percentile(all_times, 99)
    all_times = all_times[all_times <= p99]
    bins = np.histogram_bin_edges(all_times, bins=50)

    for name, df in variant_dfs.items():
        sns.histplot(df["time"], label=name, alpha=0.5, ax=ax, bins=bins)

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_rank_boxplot(
    variant_dfs,
    title,
    output_path,
    variant_filter=None,
):
    filtered_items = [
        df.assign(variant=name)
        for name, df in variant_dfs.items()
        if variant_filter is None or variant_filter in name
    ]
    if not filtered_items:
        raise ValueError("No variants available for rank boxplot")
    df_plot = pd.concat(filtered_items, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df_plot, x="rank", y="time", hue="variant", ax=ax)
    ax.set_xlabel("Query Rank")
    ax.set_ylabel("Time (μs)")
    plt.yscale("log")
    ax.set_title(title)
    plt.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_throughput_lineplot(
    variant_dfs: Dict[str, pd.DataFrame],
    title: str,
    output_path: Path,
    threads_filter: int | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))

    for variant, df in sorted(variant_dfs.items()):
        algorithm = _parse_algorithm_name(variant)
        threads_values = sorted(df["threads"].unique())
        for threads in threads_values:
            if threads_filter is not None and threads != threads_filter:
                continue
            df_subset = df[df["threads"] == threads]
            label = f"{algorithm} (threads={threads})"
            ax.plot(
                df_subset["rank"],
                df_subset["throughput"],
                marker="o",
                label=label,
                alpha=0.7,
            )

    ax.set_xlabel("Rank")
    ax.set_yscale("log")
    ax.set_ylabel("Throughput (queries/s)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_plots(graph, query, device, entry, baseline_entry, threads_filter=None):
    all_variants = {}
    if baseline_entry:
        all_variants.update(baseline_entry.dfs)
    all_variants.update(entry.dfs)

    if len(all_variants) < 1:
        click.echo(
            f"Skipping {graph}/{query}/{device}: no variant data available.",
            err=True,
        )
        return

    base_filename = f"{graph}_{query}_{device}"
    title = f"{graph} device={device}"

    throughput_dfs, time_dfs = _partition_variant_dfs(all_variants)

    if throughput_dfs:
        throughput_path = entry.directory / f"{base_filename}_throughput.png"
        _plot_throughput_lineplot(
            throughput_dfs, f"{title} - Throughput", throughput_path, threads_filter
        )
        click.echo(
            f"Saved throughput plot for graph={graph}, query={query}, device={device} -> "
            f"{throughput_path.name}"
        )

    if time_dfs:
        if len(time_dfs) >= 2 and baseline_entry is not None:
            validate_distances(time_dfs)

        histogram_path = entry.directory / f"{base_filename}_histogram.png"
        rank_path = entry.directory / f"{base_filename}_rank_boxplot.png"

        _plot_histogram(time_dfs, title, histogram_path)
        _plot_rank_boxplot(time_dfs, title, rank_path)

        click.echo(
            f"Saved plots for graph={graph}, query={query}, device={device} -> "
            f"{histogram_path.name}, {rank_path.name}"
        )


def handle(
    xp_name: str | None = None,
    device: str | None = None,
    variant: str | None = None,
    threads: int | None = None,
) -> None:
    workspace_root = get_workspace_root()
    results_base = workspace_root / "experiments" / "results"

    if xp_name:
        xp_names = [xp_name]
    else:
        xp_names = get_experiment_names(results_base)

    for name in sorted(xp_names):
        xp_path = results_base / name
        experiments = collect_experiments(
            xp_path, device_filter=device, variant_filter=variant
        )
        if not experiments:
            continue

        grouped = group_by_query(experiments)
        for (graph, query), devices in sorted(grouped.items()):
            baseline = devices.get("cpu")
            for device_id, entry in sorted(devices.items()):
                if device_id == "cpu":
                    continue
                _save_plots(graph, query, device_id, entry, baseline, threads)
