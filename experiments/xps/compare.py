import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import click
import pandas as pd
from tabulate import tabulate

from .paths import get_workspace_root


def get_experiment_data(
    xp_path: Path,
) -> Dict[Tuple[str, str, str], Dict[str, pd.DataFrame]]:
    # Discover experiment structure: {xp_base_path}/{xp_name}/{graph}/{query_hash}/{device_id}/*.csv
    experiments = {}

    if not xp_path.exists():
        click.echo(f"Error: Experiment path {xp_path} does not exist.", err=True)
        return {}

    # Scan through the hierarchy
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

                exp_key = (graph_name, query_hash, device_id)
                if exp_key not in experiments:
                    experiments[exp_key] = {}

                for csv_file in device_id_dir.glob("*.csv"):
                    # Parse filename: {timestamp}_{gitsha}_{variant}.csv
                    match = re.match(r"(\d+)_(.+)\.csv", csv_file.name)
                    if not match:
                        continue

                    timestamp = int(match.group(1))
                    variant = match.group(2)

                    if variant not in experiments[exp_key]:
                        experiments[exp_key][variant] = []
                    experiments[exp_key][variant].append((timestamp, csv_file))

    # Sort by timestamp (most recent first) and take the most recent for each algorithm
    for exp_key in experiments:
        for variant in experiments[exp_key]:
            experiments[exp_key][variant].sort(reverse=True, key=lambda x: x[0])

    # Load data for each complete experiment set
    experiment_data = {}
    for exp_key, variants in experiments.items():
        variant_dfs = {
            variant: pd.read_csv(files[0][1]) for variant, files in variants.items()
        }
        experiment_data[exp_key] = variant_dfs

    return experiment_data


def print_tables(
    exp_key: Tuple[str, str, str],
    variant_dfs: Dict[str, pd.DataFrame],
    baseline_dfs: Dict[str, pd.DataFrame] | None = None,
) -> None:
    graph_name, query_hash, device_id = exp_key

    # Merge baseline (CPU) if provided
    all_variants = baseline_dfs.copy() if baseline_dfs else {}
    all_variants.update(variant_dfs)

    if not all_variants:
        return

    click.echo(f"Graph: {graph_name}, Query: {query_hash}, Device: {device_id}")

    # Table 1: Percentiles
    percentiles = [0.01, 0.1, 0.5, 0.9, 0.99]
    perc_names = ["p1", "p10", "p50", "p90", "p99"]

    perc_data = []
    for variant, df in sorted(all_variants.items()):
        row = {"variant": variant}
        vals = df["time"].quantile(percentiles).values
        for name, val in zip(perc_names, vals):
            row[name] = f"{val:,.0f}"
        perc_data.append(row)

    perc_df = pd.DataFrame(perc_data)
    click.echo("\nExecution Time Percentiles (μs):")
    click.echo(tabulate(perc_df, headers="keys", tablefmt="github", showindex=False))

    # Table 2: Avg time per rank
    combined_df = pd.concat(
        [df.assign(variant=variant) for variant, df in all_variants.items()],
        ignore_index=True,
    )
    rank_avg = combined_df.groupby(["variant", "rank"])["time"].mean().unstack()

    # Format the rank_avg table for better readability
    rank_avg_formatted = rank_avg.map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")

    # Split the table into two halves by columns to avoid terminal overflow
    num_cols = len(rank_avg_formatted.columns)
    mid = (num_cols + 1) // 2

    first_half = rank_avg_formatted.iloc[:, :mid]
    second_half = rank_avg_formatted.iloc[:, mid:]

    click.echo("\nAverage Execution Time per Dijkstra Rank (μs):")
    click.echo(tabulate(first_half, headers="keys", tablefmt="github", showindex=True))

    if not second_half.empty:
        click.echo(
            "\n"+
            tabulate(second_half, headers="keys", tablefmt="github", showindex=True)
        )


def handle(xp_name: str | None = None, verbose: bool = True) -> str:
    """Compare experiment results and return/print summary tables."""
    workspace_root = get_workspace_root()
    results_base = workspace_root / "experiments" / "results"

    if xp_name:
        xp_names = [xp_name]
    else:
        # Try to find experiments
        if not results_base.exists():
            click.echo(f"Error: {results_base} not found.", err=True)
            sys.exit(1)
        xp_names = [d.name for d in results_base.iterdir() if d.is_dir()]
        if not xp_names:
            click.echo("No experiments found in experiments/results/", err=True)
            sys.exit(1)

    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        for name in sorted(xp_names):
            xp_path = results_base / name
            data = get_experiment_data(xp_path)

            # Group by (graph, query) to find baseline
            grouped_data = {}
            for (graph, query, device), variants in data.items():
                key = (graph, query)
                if key not in grouped_data:
                    grouped_data[key] = {}
                grouped_data[key][device] = variants

            for (graph, query), devices in sorted(grouped_data.items()):
                baseline_dfs = devices.get("cpu", {})
                for device, variants in sorted(devices.items()):
                    if device == "cpu":
                        continue
                    print_tables((graph, query, device), variants, baseline_dfs)

    final_output = output.getvalue()
    if verbose:
        click.echo(final_output, nl=False)
    return final_output
