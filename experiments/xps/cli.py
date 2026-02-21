import click

from . import add as add_cmd
from . import create as create_cmd
from . import run as run_cmd
from . import update as update_cmd


@click.group()
def main() -> None:
    """Unified experiment management tool for GPUSSSP."""


@main.command()
@click.argument("xp_name")
def create(xp_name: str) -> None:
    """Create a new experiment branch."""

    create_cmd.handle(xp_name)


@main.command()
@click.argument("run_targets")
@click.argument("params")
def add(run_targets: str, params: str) -> None:
    """Add an instrumentation commit to the current experiment branch."""

    add_cmd.handle(run_targets, params)


@main.command()
def run() -> None:
    """Run all experiments in the current branch."""

    run_cmd.handle()


@main.command()
@click.argument("xp_name", required=False)
def update(xp_name: str | None) -> None:
    """Rebase experiment branch on main and drop result commits."""

    update_cmd.handle(xp_name)


@main.command(name="_rebase", hidden=True)
@click.argument("xp_name")
@click.argument("todo_file")
def rebase_helper(xp_name: str, todo_file: str) -> None:
    """Hidden helper for interactive rebase to drop result commits."""

    update_cmd.rebase_helper(xp_name, todo_file)
