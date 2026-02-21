import os
import re
from pathlib import Path
from typing import Optional

import click

from .errors import error_exit
from .gitops import extract_xp_name, get_current_branch, run_command
from .paths import get_launcher_path


def rebase_helper(xp_name: str, todo_file_path: str) -> None:
    todo_file = Path(todo_file_path)

    if not todo_file.exists():
        error_exit(f"Rebase todo file not found: {todo_file}")

    lines = todo_file.read_text().splitlines()
    new_lines = []
    # Matches all result commits we want to drop in the rebase
    pattern = re.compile(rf"^pick\s+([a-f0-9]+)\s+#\s+result:xp:{xp_name}")

    for line in lines:
        if pattern.match(line):
            new_lines.append(line.replace("pick", "drop", 1))
        else:
            new_lines.append(line)

    todo_file.write_text("\n".join(new_lines) + "\n")


def handle(xp_name_arg: Optional[str]) -> None:
    current_branch = get_current_branch()

    if xp_name_arg:
        xp_name = xp_name_arg
        branch_name = f"experiment/{xp_name}"
    else:
        if not current_branch.startswith("experiment/"):
            error_exit("Not on an experiment branch and no experiment name provided.")
        branch_name = current_branch
        xp_name = extract_xp_name(branch_name)

    if current_branch != branch_name:
        click.echo(f"Switching to branch: {branch_name}")
        run_command(["git", "checkout", branch_name])

    click.echo(f"Rebasing {branch_name} on origin/main and dropping result commits...")

    env = os.environ.copy()
    script_path = get_launcher_path()
    env["GIT_SEQUENCE_EDITOR"] = f'"{script_path}" _rebase {xp_name}'

    exit_code, _, stderr = run_command(
        ["git", "rebase", "-i", "origin/main"], check=False, env=env
    )

    if exit_code != 0:
        click.echo(
            f"Rebase failed. Please resolve conflicts manually.\n{stderr}", err=True
        )
        raise SystemExit(exit_code)

    click.echo("\n" + "=" * 80)
    click.echo(f"✓ Experiment branch '{branch_name}' updated and cleaned!")
    click.echo("=" * 80)
    click.echo()
