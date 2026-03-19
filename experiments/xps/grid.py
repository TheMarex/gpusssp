import re
from pathlib import Path
from typing import Dict, Tuple

import click
import pandas as pd
from tabulate import tabulate

from .discovery import (
    ExperimentEntry,
    collect_experiments,
    get_experiment_names,
    group_by_query,
)
from .paths import get_workspace_root


def parse_variant_params(
    variant: str, param1: str, param2: str
) -> Tuple[str | None, str | None]:
    pattern1 = rf"{param1}(\d+)"
    pattern2 = rf"{param2}(\d+)"

    match1 = re.search(pattern1, variant)
    match2 = re.search(pattern2, variant)

    val1 = match1.group(1) if match1 else None
    val2 = match2.group(1) if match2 else None

    return val1, val2


def collect_timing_data(
    variant_dfs: Dict[str, pd.DataFrame],
    param1: str,
    param2: str,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    timing_data: Dict[Tuple[str, str], Dict[int, float]] = {}

    for variant, df in variant_dfs.items():
        val1, val2 = parse_variant_params(variant, param1, param2)
        if val1 is None or val2 is None:
            continue

        key = (val1, val2)
        if key not in timing_data:
            timing_data[key] = {}

        rank_avg = df.groupby("rank")["time"].mean()
        for rank, avg_time in rank_avg.items():
            timing_data[key][int(rank)] = avg_time

    return timing_data


def build_best_param2_grid(
    timing_data: Dict[Tuple[str, str], Dict[int, float]],
    param1: str,
    param2: str,
) -> pd.DataFrame | None:
    rows_data: Dict[str, Dict[int, Tuple[str, float]]] = {}

    for (val1, val2), rank_times in timing_data.items():
        if val1 not in rows_data:
            rows_data[val1] = {}

        for rank, avg_time in rank_times.items():
            if rank not in rows_data[val1]:
                rows_data[val1][rank] = (val2, avg_time)
            elif avg_time < rows_data[val1][rank][1]:
                rows_data[val1][rank] = (val2, avg_time)

    if not rows_data:
        return None

    all_ranks = set()
    for rank_dict in rows_data.values():
        all_ranks.update(rank_dict.keys())
    sorted_ranks = sorted(all_ranks)

    sorted_param1_vals = sorted(rows_data.keys(), key=lambda x: int(x))

    grid_data = []
    for p1_val in sorted_param1_vals:
        row = {param1: p1_val}
        for rank in sorted_ranks:
            if rank in rows_data[p1_val]:
                row[rank] = rows_data[p1_val][rank][0]
            else:
                row[rank] = "-"
        grid_data.append(row)

    return pd.DataFrame(grid_data)


def find_best_per_rank(
    timing_data: Dict[Tuple[str, str], Dict[int, float]],
) -> Dict[int, float]:
    best_per_rank: Dict[int, float] = {}

    for rank_times in timing_data.values():
        for rank, avg_time in rank_times.items():
            if rank not in best_per_rank or avg_time < best_per_rank[rank]:
                best_per_rank[rank] = avg_time

    return best_per_rank


def find_winning_combos(
    timing_data: Dict[Tuple[str, str], Dict[int, float]],
    best_per_rank: Dict[int, float],
) -> set[Tuple[str, str]]:
    winners = set()

    for (val1, val2), rank_times in timing_data.items():
        for rank, avg_time in rank_times.items():
            if avg_time == best_per_rank[rank]:
                winners.add((val1, val2))
                break

    return winners


def find_top3_combos(
    timing_data: Dict[Tuple[str, str], Dict[int, float]],
    best_per_rank: Dict[int, float],
) -> set[Tuple[str, str]]:
    top3_combos = set()

    for rank in best_per_rank.keys():
        rank_times = []
        for (val1, val2), times in timing_data.items():
            if rank in times:
                rank_times.append((times[rank], (val1, val2)))

        rank_times.sort(key=lambda x: x[0])
        for _, combo in rank_times[:3]:
            top3_combos.add(combo)

    return top3_combos


def get_combos_for_show(
    show: str,
    timing_data: Dict[Tuple[str, str], Dict[int, float]],
    best_per_rank: Dict[int, float],
) -> set[Tuple[str, str]]:
    if show == "all":
        return set(timing_data.keys())
    elif show == "top3":
        return find_top3_combos(timing_data, best_per_rank)
    else:
        return find_winning_combos(timing_data, best_per_rank)


def build_speedup_grid(
    timing_data: Dict[Tuple[str, str], Dict[int, float]],
    best_per_rank: Dict[int, float],
    combos: set[Tuple[str, str]],
    param1: str,
    param2: str,
) -> pd.DataFrame | None:
    if not combos:
        return None

    all_ranks = sorted(best_per_rank.keys())

    sorted_combos = sorted(
        combos,
        key=lambda x: (int(x[0]), int(x[1])),
    )

    grid_data = []
    for val1, val2 in sorted_combos:
        row = {param1: val1, param2: val2}
        for rank in all_ranks:
            if rank in timing_data[(val1, val2)]:
                this_time = timing_data[(val1, val2)][rank]
                best_time = best_per_rank[rank]
                speedup = best_time / this_time
                row[rank] = f"{speedup:.2f}"
            else:
                row[rank] = "-"
        grid_data.append(row)

    return pd.DataFrame(grid_data)


def print_grids(
    exp_key: Tuple[str, str, str],
    variant_dfs: Dict[str, pd.DataFrame],
    param1: str,
    param2: str,
    show: str,
) -> None:
    graph_name, query_hash, device_id = exp_key

    timing_data = collect_timing_data(variant_dfs, param1, param2)
    if not timing_data:
        click.echo(
            f"No matching variants found for {graph_name}/{query_hash}/{device_id} "
            f"with params {param1}, {param2}"
        )
        return

    click.echo(f"\nGraph: {graph_name}, Query: {query_hash}, Device: {device_id}")

    grid1 = build_best_param2_grid(timing_data, param1, param2)
    if grid1 is not None and not grid1.empty:
        click.echo(f"\nGrid 1: {param1} (rows) vs rank (cols), showing best {param2}")
        click.echo(tabulate(grid1, headers="keys", tablefmt="github", showindex=False))

    best_per_rank = find_best_per_rank(timing_data)
    combos = get_combos_for_show(show, timing_data, best_per_rank)

    grid2 = build_speedup_grid(timing_data, best_per_rank, combos, param1, param2)
    if grid2 is not None and not grid2.empty:
        show_desc = {
            "winners": "Winning combos",
            "top3": "Top 3 combos",
            "all": "All combos",
        }
        click.echo(
            f"\nGrid 2: {show_desc.get(show, show)} showing speedup vs best per rank (1.0 = winner)"
        )
        click.echo(tabulate(grid2, headers="keys", tablefmt="github", showindex=False))


def handle(
    xp_name: str | None = None,
    device: str | None = None,
    variant: str | None = None,
    param1: str = "delta",
    param2: str = "batch",
    show: str = "winners",
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
            for device_id, entry in sorted(devices.items()):
                if device_id == "cpu":
                    continue
                print_grids((graph, query, device_id), entry.dfs, param1, param2, show)
