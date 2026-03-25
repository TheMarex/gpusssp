import math

import click

from .errors import error_exit
from .gitops import extract_xp_name, get_current_branch, run_command
from .runner import build_commit_message
from .validation import (
    parse_metrics,
    parse_params,
    validate_metrics,
    validate_params,
    validate_run_targets,
)


def handle(run_targets: str, params_str: str, metrics_str: str = "") -> None:
    current_branch = get_current_branch()

    if not current_branch.startswith("experiment/"):
        error_exit(
            f"Not on an experiment branch. Current branch: {current_branch}\n"
            "Use 'xps.py create <xp_name>' to create one."
        )

    xp_name = extract_xp_name(current_branch)

    targets = [t.strip() for t in run_targets.split(",")]
    params = parse_params(params_str)
    metrics = parse_metrics(metrics_str)

    validate_run_targets(targets)
    if "throughput" in targets and len([t for t in targets if t != "throughput"]) == 0:
        error_exit(
            "throughput target requires at least one other algorithm target (e.g., nearfar, dijkstra, dial)"
        )
    validate_params(params)
    validate_metrics(metrics)

    num_combinations = math.prod(len(v.split(",")) for v in params.values())

    xp_commit_msg = build_commit_message(xp_name, targets, params, metrics)

    click.echo(f"Adding instrumentation commit to branch: {current_branch}")
    click.echo(f"Commit message: {xp_commit_msg}")
    click.echo()

    run_command(["git", "commit", "--allow-empty", "-m", xp_commit_msg])

    click.echo("=" * 80)
    click.echo("✓ Instrumentation commit created!")
    click.echo("=" * 80)
    click.echo(f"Experiment: {xp_name}")
    click.echo(f"Run targets: {', '.join(targets)}")
    click.echo(f"Parameters: {' '.join(f'{k}={v}' for k, v in params.items())}")
    click.echo(f"Combinations: {num_combinations}")
    click.echo(f"Metrics: {', '.join(metrics)}")
    click.echo()
    click.echo("Next steps:")
    click.echo('  - Add more variants: xps.py add "<runs>" "<params>" ["<metrics>"]')
    click.echo("  - Run experiments: xps.py run")
    click.echo()
