from typing import Dict, List, Tuple

import click
import pandas as pd
from tabulate import tabulate

from .discovery import collect_experiments, get_experiment_names, group_by_query
from .paths import get_workspace_root


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
    click.echo(tabulate(perc_df, headers="keys", tablefmt="github", showindex=False))

    combined_df = pd.concat(
        [df.assign(variant=variant) for variant, df in all_variants.items()],
        ignore_index=True,
    )
    rank_avg = combined_df.groupby(["variant", "rank"])[metric].mean().unstack()

    rank_avg_formatted = rank_avg.map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")

    num_cols = len(rank_avg_formatted.columns)
    mid = (num_cols + 1) // 2

    first_half = rank_avg_formatted.iloc[:, :mid]
    second_half = rank_avg_formatted.iloc[:, mid:]

    click.echo(f"\nAverage {metric.capitalize()} per Dijkstra Rank:")
    click.echo(tabulate(first_half, headers="keys", tablefmt="github", showindex=True))

    if not second_half.empty:
        click.echo(
            "\n"
            + tabulate(second_half, headers="keys", tablefmt="github", showindex=True)
        )

    winners_per_rank = rank_avg.idxmin(axis=0).dropna()
    unique_winners = winners_per_rank.unique()

    if not winners_per_rank.empty:
        click.echo("\nWinners:")

        def get_speedup_df(winner: str, ranks: pd.Index) -> pd.DataFrame:
            speedup_data = {}
            for v in sorted(all_variants.keys()):
                if v == winner:
                    continue
                speedups = []
                for r in ranks:
                    val = rank_avg.loc[v, r] / rank_avg.loc[winner, r]
                    speedups.append(f"{val:.2f}x")
                speedup_data[v] = speedups
            return pd.DataFrame(speedup_data, index=ranks).T

        if len(unique_winners) == 1:
            winner = unique_winners[0]
            click.echo(f"\nWinner (all ranks): {winner}")
            df = get_speedup_df(winner, rank_avg.columns)
            click.echo(tabulate(df, headers="keys", tablefmt="github", showindex=True))
        else:
            w_start = winners_per_rank.iloc[0]
            m = 0
            while (
                m + 1 < len(winners_per_rank)
                and winners_per_rank.iloc[m + 1] == w_start
            ):
                m += 1

            w_end = winners_per_rank.iloc[-1]
            k = len(winners_per_rank) - 1
            while k - 1 >= 0 and winners_per_rank.iloc[k - 1] == w_end:
                k -= 1

            start_ranks = winners_per_rank.index[0 : m + 1]
            end_ranks = winners_per_rank.index[k:]

            click.echo(f"\nSpeedup of winner {w_start}")
            df_start = get_speedup_df(w_start, start_ranks)
            click.echo(
                tabulate(df_start, headers="keys", tablefmt="github", showindex=True)
            )

            click.echo(f"\nSpeedup of winner {w_end}")
            df_end = get_speedup_df(w_end, end_ranks)
            click.echo(
                tabulate(df_end, headers="keys", tablefmt="github", showindex=True)
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
