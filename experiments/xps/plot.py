import sys
from typing import Dict, Tuple

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


def _save_plots(graph, query, device, entry, baseline_entry):
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

    if len(all_variants) >= 2 and baseline_entry is not None:
        validate_distances(all_variants)

    base_filename = f"{graph}_{query}_{device}"
    histogram_path = entry.directory / f"{base_filename}_histogram.png"
    rank_path = entry.directory / f"{base_filename}_rank_boxplot.png"

    title = f"{graph} device={device}"

    _plot_histogram(
        all_variants,
        title,
        histogram_path,
    )
    _plot_rank_boxplot(
        all_variants,
        title,
        rank_path,
    )

    click.echo(
        f"Saved plots for graph={graph}, query={query}, device={device} -> "
        f"{histogram_path.name}, {rank_path.name}"
    )


def handle(
    xp_name: str | None = None,
    device: str | None = None,
    variant: str | None = None,
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
                # If we filtered by device in collect_experiments, it should only contain the device and cpu
                # So if device is set and device_id != device, it shouldn't be here anyway.
                _save_plots(graph, query, device_id, entry, baseline)
