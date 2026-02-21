import click

from .errors import error_exit
from .gitops import get_current_branch, run_command
from .validation import validate_xp_name


def handle(xp_name: str) -> None:
    validate_xp_name(xp_name)

    branch_name = f"experiment/{xp_name}"
    current_branch = get_current_branch()

    exit_code, _, _ = run_command(
        ["git", "rev-parse", "--verify", branch_name], check=False
    )
    if exit_code == 0:
        error_exit(f"Branch '{branch_name}' already exists")

    click.echo(f"Creating experiment branch: {branch_name}")
    click.echo(f"From branch: {current_branch}")

    run_command(["git", "checkout", "-b", branch_name])

    click.echo()
    click.echo("=" * 80)
    click.echo(f"✓ Experiment branch '{branch_name}' created!")
    click.echo("=" * 80)
    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Make code changes (if needed) and commit them")
    click.echo('  2. Add instrumentation: xps.py add "<runs>" "<params>"')
    click.echo("  3. Run experiments: xps.py run")
    click.echo()
