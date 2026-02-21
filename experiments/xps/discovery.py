import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import click
import pandas as pd


@dataclass
class ExperimentEntry:
    dfs: Dict[str, pd.DataFrame]
    directory: Path


def get_experiment_names(results_base: Path) -> List[str]:
    """Find all experiments in the results base directory."""
    if not results_base.exists():
        click.echo(f"Error: {results_base} not found.", err=True)
        sys.exit(1)

    xp_names = [d.name for d in results_base.iterdir() if d.is_dir()]
    if not xp_names:
        click.echo("No experiments found in experiments/results/", err=True)
        sys.exit(1)

    return sorted(xp_names)


def collect_experiments(
    xp_path: Path,
    device_filter: str | None = None,
    variant_filter: str | None = None,
) -> Dict[Tuple[str, str, str], ExperimentEntry]:
    """Collect the most recent experiment results from a path."""
    experiments = {}

    if not xp_path.exists():
        click.echo(f"Error: Experiment path {xp_path} does not exist.", err=True)
        return {}

    # Scan through the hierarchy: {xp_path}/{graph}/{query_hash}/{device_id}/*.csv
    for graph_dir in xp_path.iterdir():
        if not graph_dir.is_dir():
            continue
        graph_name = graph_dir.name

        for query_hash_dir in graph_dir.iterdir():
            if not query_hash_dir.is_dir():
                continue
            query_hash = query_hash_dir.name

            for device_id_dir in query_hash_dir.iterdir():
                if not device_id_dir.is_dir():
                    continue
                device_id = device_id_dir.name

                if device_filter and device_id != device_filter and device_id != "cpu":
                    continue

                exp_key = (graph_name, query_hash, device_id)
                entry = experiments.setdefault(
                    exp_key, {"files": {}, "dir": device_id_dir}
                )

                for csv_file in device_id_dir.glob("*.csv"):
                    # Parse filename: {timestamp}_{gitsha}_{variant}.csv
                    match = re.match(r"(\d+)_(.+)\.csv", csv_file.name)
                    if not match:
                        continue

                    # timestamp = int(match.group(1))
                    variant = match.group(2)

                    if variant_filter and variant_filter not in variant:
                        continue

                    timestamp = int(match.group(1))
                    entry["files"].setdefault(variant, []).append((timestamp, csv_file))

    # Load data for the most recent variant of each experiment
    loaded = {}
    for exp_key, entry in experiments.items():
        variant_files = entry["files"]
        if not variant_files:
            continue

        for variant in variant_files:
            # Sort by timestamp (most recent first)
            variant_files[variant].sort(reverse=True, key=lambda item: item[0])

        variant_dfs = {
            variant: pd.read_csv(files[0][1])
            for variant, files in variant_files.items()
        }
        loaded[exp_key] = ExperimentEntry(dfs=variant_dfs, directory=entry["dir"])

    return loaded


def group_by_query(
    experiments: Dict[Tuple[str, str, str], ExperimentEntry],
) -> Dict[Tuple[str, str], Dict[str, ExperimentEntry]]:
    """Group experiments by (graph, query_hash)."""
    grouped = {}
    for (graph, query, device), entry in experiments.items():
        grouped.setdefault((graph, query), {})[device] = entry
    return grouped
