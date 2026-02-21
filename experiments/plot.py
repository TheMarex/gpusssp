#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "click", "matplotlib", "seaborn", "numpy"]
# ///
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, Tuple

import click
import pandas as pd

from plots import plot_histogram, plot_rank_boxplot
from evaluation import validate_distances


RESULTS_BASE = Path("experiments/results")


@dataclass
class ExperimentEntry:
    dfs: Dict[str, pd.DataFrame]
    directory: Path


def _collect_experiments(xp_path: Path) -> Dict[Tuple[str, str, str], ExperimentEntry]:
    experiments = {}

    if not xp_path.exists():
        click.echo(f"Error: Experiment path {xp_path} does not exist.", err=True)
        return {}

    for graph_dir in xp_path.iterdir():
        if not graph_dir.is_dir():
            continue
        graph_name = graph_dir.name

        for query_hash_dir in graph_dir.iterdir():
            if not query_hash_dir.is_dir():
                continue
            query_hash = query_hash_dir.name

            for device_dir in query_hash_dir.iterdir():
                if not device_dir.is_dir():
                    continue
                device_id = device_dir.name

                exp_key = (graph_name, query_hash, device_id)
                entry = experiments.setdefault(
                    exp_key, {"files": {}, "dir": device_dir}
                )

                for csv_file in device_dir.glob("*.csv"):
                    match = re.match(r"(\d+)_(.+)\.csv", csv_file.name)
                    if not match:
                        continue
                    timestamp = int(match.group(1))
                    variant = match.group(2)
                    entry["files"].setdefault(variant, []).append((timestamp, csv_file))

    loaded = {}
    for exp_key, entry in experiments.items():
        variant_files = entry["files"]
        for variant in variant_files:
            variant_files[variant].sort(reverse=True, key=lambda item: item[0])

        variant_dfs = {
            variant: pd.read_csv(files[0][1])
            for variant, files in variant_files.items()
        }
        loaded[exp_key] = ExperimentEntry(dfs=variant_dfs, directory=entry["dir"])

    return loaded


def _group_by_query(experiments):
    grouped = {}
    for (graph, query, device), entry in experiments.items():
        grouped.setdefault((graph, query), {})[device] = entry
    return grouped


def _save_plots(graph, query, device, entry, baseline_entry, show):
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

    plot_histogram(
        all_variants,
        title,
        output_path=histogram_path,
        show=show,
    )
    plot_rank_boxplot(
        all_variants,
        title,
        output_path=rank_path,
        show=show,
    )

    click.echo(
        f"Saved plots for graph={graph}, query={query}, device={device} -> "
        f"{histogram_path.name}, {rank_path.name}"
    )


@click.command()
@click.argument("xp_name", required=False)
@click.option(
    "--show", is_flag=True, help="Display plots interactively in addition to saving."
)
def main(xp_name, show):
    if xp_name:
        xp_names = [xp_name]
    else:
        if not RESULTS_BASE.exists():
            click.echo(f"Error: {RESULTS_BASE} not found.", err=True)
            sys.exit(1)
        xp_names = [d.name for d in RESULTS_BASE.iterdir() if d.is_dir()]
        if not xp_names:
            click.echo("No experiments found in experiments/results/", err=True)
            sys.exit(1)

    for name in sorted(xp_names):
        xp_path = RESULTS_BASE / name
        experiments = _collect_experiments(xp_path)
        if not experiments:
            continue

        grouped = _group_by_query(experiments)
        for (graph, query), devices in sorted(grouped.items()):
            baseline = devices.get("cpu")
            for device, entry in sorted(devices.items()):
                if device == "cpu":
                    continue
                _save_plots(graph, query, device, entry, baseline, show)


if __name__ == "__main__":
    main()
