import click

from .errors import error_exit
from .gitops import extract_xp_name, get_current_branch, run_command
from .runner import build_commit_message
from .validation import parse_params, validate_params, validate_run_targets


def handle(run_targets: str, params_str: str) -> None:
    current_branch = get_current_branch()

    if not current_branch.startswith("experiment/"):
        error_exit(
            f"Not on an experiment branch. Current branch: {current_branch}\n"
            "Use 'xps.py create <xp_name>' to create one."
        )

    xp_name = extract_xp_name(current_branch)

    targets = [t.strip() for t in run_targets.split(",")]
    params = parse_params(params_str)

    validate_run_targets(targets)
    validate_params(params)

    xp_commit_msg = build_commit_message(xp_name, targets, params)

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
    click.echo()
    click.echo("Next steps:")
    click.echo('  - Add more variants: xps.py add "<runs>" "<params>"')
    click.echo("  - Run experiments: xps.py run")
    click.echo()
