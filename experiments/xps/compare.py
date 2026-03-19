from typing import Dict, List, Tuple

import click
import pandas as pd
from tabulate import tabulate

from .discovery import collect_experiments, get_experiment_names, group_by_query
from .paths import get_workspace_root

def print_split_table(
    df: pd.DataFrame, header: str, showindex: bool = True
) -> None:
    num_cols = len(df.columns)
    mid = (num_cols + 1) // 2
    first_half = df.iloc[:, :mid]
    second_half = df.iloc[:, mid:]

    click.echo(f"\n{header}")
    click.echo(
        tabulate(first_half, headers="keys", tablefmt="github", showindex=showindex)
    )

    if not second_half.empty:
        click.echo(
            "\n"
            + tabulate(
                second_half, headers="keys", tablefmt="github", showindex=showindex
            )
        )

def print_tables(
    exp_key: Tuple[str, str, str],
    variant_dfs: Dict[str, pd.DataFrame],
    metric: str,
    baseline_dfs: Dict[str, pd.DataFrame] | None = None,
) -> None:
    graph_name, query_hash, device_id = exp_key

    all_variants = baseline_dfs.copy() if baseline_dfs else {}
    all_variants.update(variant_dfs)

    all_variants = {v: df for v, df in all_variants.items() if metric in df.columns}

    if not all_variants:
        return

    click.echo(f"Graph: {graph_name}, Query: {query_hash}, Device: {device_id}")
    click.echo(f"Metric: {metric}")

    has_ranks = any(
        "rank" in df.columns and df["rank"].nunique() > 1
        for df in all_variants.values()
    )

    if not has_ranks:
        percentiles = [0.01, 0.1, 0.5, 0.9, 0.99]
        perc_names = ["p1", "p10", "p50", "p90", "p99"]

        perc_data = []
        for variant, df in sorted(all_variants.items()):
            row = {"variant": variant}
            vals = df[metric].quantile(percentiles).values
            for name, val in zip(perc_names, vals):
                row[name] = f"{val:,.0f}"
            perc_data.append(row)

        perc_df = pd.DataFrame(perc_data)
        click.echo(f"\n{metric.capitalize()} Percentiles:")
        click.echo(
            tabulate(perc_df, headers="keys", tablefmt="github", showindex=False)
        )
        return

    combined_df = pd.concat(
        [df.assign(variant=variant) for variant, df in all_variants.items()],
        ignore_index=True,
    )
    rank_avg = combined_df.groupby(["variant", "rank"])[metric].mean().unstack()


    rank_avg_formatted = rank_avg.map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
    print_split_table(
        rank_avg_formatted, f"Average {metric.capitalize()} per Dijkstra Rank:"
    )

    min_per_rank = rank_avg.min(axis=0)
    normalized = rank_avg.div(min_per_rank, axis=1)
    normalized_formatted = normalized.map(
        lambda x: f"{x:.2f}" if pd.notnull(x) else "-"
    )
    print_split_table(
        normalized_formatted,
        f"Normalized {metric.capitalize()} per Dijkstra Rank (1.0 = best):"
    )

    threshold = 1.01
    winners_by_variant: Dict[str, List[int]] = {}
    for rank in rank_avg.columns:
        rank_vals = rank_avg[rank]
        min_val = rank_vals.min()
        for variant in rank_vals.index:
            if pd.notnull(rank_vals[variant]):
                ratio = rank_vals[variant] / min_val
                if ratio <= threshold:
                    if variant not in winners_by_variant:
                        winners_by_variant[variant] = []
                    winners_by_variant[variant].append(int(rank))

    if winners_by_variant:
        winner_rows = []
        for variant in sorted(winners_by_variant.keys()):
            ranks = winners_by_variant[variant]
            winner_rows.append(
                {"variant": variant, "winning_ranks": ", ".join(map(str, ranks))}
            )
        winner_df = pd.DataFrame(winner_rows)
        click.echo("\nWinners by Rank (within 1% of best):")
        click.echo(
            tabulate(winner_df, headers="keys", tablefmt="github", showindex=False)
        )


def handle(
    xp_name: str | None = None,
    device: str | None = None,
    variant: str | None = None,
    metrics: List[str] | None = None,
    verbose: bool = True,
) -> str:
    """Compare experiment results and return/print summary tables."""
    if metrics is None:
        metrics = ["time"]

    workspace_root = get_workspace_root()
    results_base = workspace_root / "experiments" / "results"

    if xp_name:
        xp_names = [xp_name]
    else:
        xp_names = get_experiment_names(results_base)

    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    with redirect_stdout(output):
        for name in sorted(xp_names):
            xp_path = results_base / name
            experiments = collect_experiments(
                xp_path, device_filter=device, variant_filter=variant
            )
            if not experiments:
                continue

            grouped = group_by_query(experiments)
            for (graph, query), devices in sorted(grouped.items()):
                baseline_entry = devices.get("cpu")
                baseline_dfs = baseline_entry.dfs if baseline_entry else {}

                non_baseline = False
                for device_id, entry in sorted(devices.items()):
                    if device_id == "cpu":
                        continue
                    non_baseline = True
                    for metric in metrics:
                        print_tables(
                            (graph, query, device_id), entry.dfs, metric, baseline_dfs
                        )

                if not non_baseline:
                    for metric in metrics:
                        print_tables((graph, query, "cpu"), baseline_dfs, metric, {})

    final_output = output.getvalue()
    if verbose:
        click.echo(final_output, nl=False)
    return final_output
