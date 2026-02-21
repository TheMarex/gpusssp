from pathlib import Path

import click

from .errors import error_exit
from .gitops import (
    extract_xp_name,
    get_current_branch,
    get_recent_commits,
    run_command,
)
from .paths import get_workspace_root
from .runner import (
    build_experiment_binary,
    collect_experiment_configs,
    ensure_release_build,
    run_experiment,
    stage_and_commit_results,
)


def handle() -> None:
    workspace_root = get_workspace_root()
    build_dir = workspace_root / "build"

    original_branch = get_current_branch()
    xp_name = extract_xp_name(original_branch)
    click.echo(f"Running experiments for: {xp_name}")

    commits = get_recent_commits(100)
    experiment_configs = collect_experiment_configs(commits, xp_name)

    if not experiment_configs:
        error_exit(f"No commits found matching 'xp:{xp_name}'")

    experiment_configs.reverse()
    click.echo(f"\nFound {len(experiment_configs)} experiment commit(s) to process\n")

    compare_script = workspace_root / "experiments" / "compare.py"

    for i, config in enumerate(experiment_configs, 1):
        click.echo(f"\n{'=' * 80}")
        click.echo(f"{xp_name} {i}/{len(experiment_configs)}: {config.commit_sha[:9]}")
        click.echo(f"{'=' * 80}\n")

        click.echo(f"Checking out {config.commit_sha[:9]}...")
        run_command(["git", "checkout", config.commit_sha])

        ensure_release_build(build_dir, workspace_root)

        for target in config.run_targets:
            build_experiment_binary(target, build_dir)

        for target in config.run_targets:
            run_experiment(target, config, workspace_root, build_dir)

        _, stdout, _ = run_command(
            [str(compare_script), xp_name], cwd=workspace_root, check=False
        )
        if stdout:
            click.echo(f"\nIntermediate results for {xp_name}:")
            click.echo(stdout)

    click.echo(f"\n{'=' * 80}")
    click.echo("All experiments completed successfully!")
    click.echo(f"{'=' * 80}\n")
    click.echo(f"Returning to branch: {original_branch}")
    run_command(["git", "checkout", original_branch])

    _, stdout, stderr = run_command(
        [str(compare_script), xp_name], cwd=workspace_root, check=False
    )

    if stdout:
        click.echo(stdout)
    if stderr:
        click.echo(stderr, err=True)

    stage_and_commit_results(workspace_root, xp_name, stdout)

    click.echo("\n✓ Experiment run complete!")
