#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# ///
"""
Unified experiment management tool for GPUSSSP.

Commands:
  create <xp_name>           Create a new experiment branch
  add <runs> <params>        Add an instrumentation commit to current experiment branch
  run                        Run all experiments in the current branch

Examples:
  # Create new experiment branch
  xps.py create delta_sweep

  # Add instrumentation commit
  xps.py add "deltastep,nearfar" "delta=900,1800,3600 data=berlin"

  # Run experiments
  xps.py run
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ExperimentConfig:
    name: str
    commit_sha: str
    run_targets: List[str]
    params: Dict[str, Any]


def get_workspace_root() -> Path:
    return Path(__file__).parent.parent.resolve()


def error_exit(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    check: bool = True,
    capture: bool = True,
) -> tuple[int, str, str]:
    """Run command and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=capture, text=True, check=False
    )

    stdout = result.stdout.strip() if capture and result.stdout else ""
    stderr = result.stderr.strip() if capture and result.stderr else ""

    if check and result.returncode != 0:
        err_msg = f"Command failed: {' '.join(cmd)}\nExit code: {result.returncode}"
        if capture and stderr:
            err_msg += f"\nStderr: {stderr}"
        error_exit(err_msg)

    return result.returncode, stdout, stderr


def get_current_branch() -> str:
    _, stdout, _ = run_command(["git", "branch", "--show-current"])
    return stdout


def extract_xp_name(branch_name: str) -> str:
    match = re.match(r"^experiment/(.+)$", branch_name)
    if not match:
        error_exit(
            f"Branch '{branch_name}' does not match pattern 'experiment/{{xp_name}}'"
        )
    return match.group(1)


def validate_xp_name(xp_name: str) -> None:
    """Validate experiment name follows conventions."""
    if not re.match(r"^[a-z0-9_]+$", xp_name):
        error_exit(
            f"Invalid experiment name '{xp_name}'. Use only lowercase letters, numbers, and underscores."
        )


def validate_run_targets(targets: List[str]) -> None:
    """Validate run targets are known experiment types."""
    valid_targets = {"dijkstra", "deltastep", "bellmanford", "nearfar"}
    for target in targets:
        if target not in valid_targets:
            error_exit(
                f"Invalid run target '{target}'. Valid targets: {', '.join(sorted(valid_targets))}"
            )


def parse_params(param_str: str) -> dict:
    """Parse param string into dictionary."""
    params = {}
    for part in param_str.split():
        if "=" not in part:
            error_exit(f"Invalid param format '{part}'. Expected key=value")
        key, value = part.split("=", 1)
        params[key] = value
    return params


def validate_params(params: dict) -> None:
    """Validate required parameters are present."""
    if "data" not in params:
        error_exit("Missing required parameter: data (e.g., data=berlin)")


def build_commit_message(xp_name: str, run_targets: List[str], params: dict) -> str:
    """Build the instrumentation commit message."""
    parts = [f"xp:{xp_name}"]

    for target in run_targets:
        parts.append(f"run:{target}")

    for key, value in params.items():
        parts.append(f"param:{key}={value}")

    return " ".join(parts)


# ============================================================================
# Command: create
# ============================================================================


def cmd_create(args: argparse.Namespace) -> None:
    """Create a new experiment branch."""
    xp_name = args.xp_name
    validate_xp_name(xp_name)

    branch_name = f"experiment/{xp_name}"
    current_branch = get_current_branch()

    # Check if branch already exists
    exit_code, _, _ = run_command(
        ["git", "rev-parse", "--verify", branch_name], check=False
    )
    if exit_code == 0:
        error_exit(f"Branch '{branch_name}' already exists")

    print(f"Creating experiment branch: {branch_name}")
    print(f"From branch: {current_branch}")

    run_command(["git", "checkout", "-b", branch_name])

    print()
    print("=" * 80)
    print(f"✓ Experiment branch '{branch_name}' created!")
    print("=" * 80)
    print()
    print("Next steps:")
    print(f"  1. Make code changes (if needed) and commit them")
    print(f'  2. Add instrumentation: xps.py add "<runs>" "<params>"')
    print(f"  3. Run experiments: xps.py run")
    print()


# ============================================================================
# Command: add
# ============================================================================


def cmd_add(args: argparse.Namespace) -> None:
    """Add an instrumentation commit to the current experiment branch."""
    current_branch = get_current_branch()

    if not current_branch.startswith("experiment/"):
        error_exit(
            f"Not on an experiment branch. Current branch: {current_branch}\n"
            "Use 'xps.py create <xp_name>' to create one."
        )

    xp_name = extract_xp_name(current_branch)

    # Parse and validate inputs
    run_targets = [t.strip() for t in args.run_targets.split(",")]
    params = parse_params(args.params)

    validate_run_targets(run_targets)
    validate_params(params)

    # Build commit message
    xp_commit_msg = build_commit_message(xp_name, run_targets, params)

    print(f"Adding instrumentation commit to branch: {current_branch}")
    print(f"Commit message: {xp_commit_msg}")
    print()

    run_command(["git", "commit", "--allow-empty", "-m", xp_commit_msg])

    print("=" * 80)
    print("✓ Instrumentation commit created!")
    print("=" * 80)
    print(f"Experiment: {xp_name}")
    print(f"Run targets: {', '.join(run_targets)}")
    print(f"Parameters: {' '.join(f'{k}={v}' for k, v in params.items())}")
    print()
    print("Next steps:")
    print('  - Add more variants: xps.py add "<runs>" "<params>"')
    print("  - Run experiments: xps.py run")
    print()


# ============================================================================
# Command: run (from run_experiments.py)
# ============================================================================


def get_recent_commits(count: int = 100) -> List[tuple[str, str]]:
    _, stdout, _ = run_command(["git", "log", f"-{count}", "--format=%H|%s"])
    commits = []
    for line in stdout.split("\n"):
        if "|" in line:
            sha, message = line.split("|", 1)
            commits.append((sha, message))
    return commits


def parse_commit_message(message: str, xp_name: str) -> Optional[ExperimentConfig]:
    expected_prefix = f"xp:{xp_name}"
    if not message.startswith(expected_prefix):
        return None

    run_targets = []
    params = {}

    for match in re.finditer(r"run:(\w+)", message):
        run_targets.append(match.group(1))

    for match in re.finditer(r"param:(\w+)=([\w,]+)", message):
        param_name = match.group(1)
        param_value = match.group(2)

        if param_name == "delta":
            params["delta"] = [int(d) for d in param_value.split(",")]
        elif param_name == "gpu":
            params["gpu"] = int(param_value)
        else:
            params[param_name] = param_value

    if not run_targets:
        error_exit(f"No run targets specified in commit message: {message}")
    if "data" not in params:
        error_exit(f"No param:data specified in commit message: {message}")

    return ExperimentConfig(
        name=xp_name, commit_sha="", run_targets=run_targets, params=params
    )


def build_experiment_binary(target: str, build_dir: Path) -> None:
    print(f"Building experiment_{target}...")
    run_command(
        ["cmake", "--build", str(build_dir), "--target", f"experiment_{target}"],
        cwd=build_dir.parent,
    )


def format_cmd(
    build_dir: Path,
    target: str,
    cache_path: Path,
    xp_base_path: Path,
    xp_name: str,
    params: Dict[str, Any],
) -> List[List[str]]:
    binary = build_dir / f"experiment_{target}"
    base_cmd = [str(binary), str(cache_path), str(xp_base_path)]

    if target in ["deltastep", "nearfar"] and "delta" in params:
        cmds = []
        for delta in params["delta"]:
            cmds.append(base_cmd + [xp_name, str(delta)])
        return cmds
    else:
        return [base_cmd + [xp_name]]


def run_experiment(
    target: str, config: ExperimentConfig, workspace_root: Path, build_dir: Path
) -> None:
    cache_path = workspace_root / "cache" / config.params["data"]
    xp_base_path = workspace_root / "experiments" / "results"

    env = os.environ.copy()
    if "gpu" in config.params:
        env["GPUSSSP_GPU"] = str(config.params["gpu"])

    cmds = format_cmd(
        build_dir, target, cache_path, xp_base_path, config.name, config.params
    )

    for cmd in cmds:
        cmd_str = " ".join(cmd)
        print(f"Running '{cmd_str}'...")
        run_command(cmd, cwd=build_dir, env=env, capture=False)


def ensure_release_build(build_dir: Path, workspace_root: Path) -> None:
    print("Configuring build for Release mode...")
    run_command(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=workspace_root,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Run all experiments in the current branch."""
    workspace_root = get_workspace_root()
    build_dir = workspace_root / "build"

    if not (workspace_root / ".git").exists():
        error_exit("Not in a git repository")

    original_branch = get_current_branch()
    xp_name = extract_xp_name(original_branch)
    print(f"Running experiments for: {xp_name}")

    commits = get_recent_commits(100)
    experiment_configs = []

    for sha, message in commits:
        config = parse_commit_message(message, xp_name)
        if config:
            config.commit_sha = sha
            experiment_configs.append(config)

    if not experiment_configs:
        error_exit(f"No commits found matching 'xp:{xp_name}'")

    experiment_configs.reverse()
    print(f"\nFound {len(experiment_configs)} experiment commit(s) to process\n")

    for i, config in enumerate(experiment_configs, 1):
        print(f"\n{'=' * 80}")
        print(f"{xp_name} {i}/{len(experiment_configs)}: {config.commit_sha[:9]}")
        print(f"{'=' * 80}\n")

        print(f"Checking out {config.commit_sha[:9]}...")
        run_command(["git", "checkout", config.commit_sha])

        ensure_release_build(build_dir, workspace_root)

        for target in config.run_targets:
            build_experiment_binary(target, build_dir)

        for target in config.run_targets:
            run_experiment(target, config, workspace_root, build_dir)

        # Print intermediate results
        compare_script = workspace_root / "experiments" / "compare.py"
        _, stdout, _ = run_command(
            [str(compare_script), xp_name], cwd=workspace_root, check=False
        )
        if stdout:
            print(f"\nIntermediate results for {xp_name}:")
            print(stdout)

    print(f"\n{'=' * 80}")
    print("All experiments completed successfully!")
    print(f"{'=' * 80}\n")
    print(f"Returning to branch: {original_branch}")
    run_command(["git", "checkout", original_branch])

    results_dir = workspace_root / "experiments" / "results" / xp_name
    if results_dir.exists():
        print(f"\nComparing results...")
        compare_script = workspace_root / "experiments" / "compare.py"
        _, stdout, stderr = run_command(
            [str(compare_script), xp_name], cwd=workspace_root, check=False
        )

        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)

        print(f"\nCommitting results from {results_dir}...")
        run_command(["git", "add", "-f", str(results_dir)])

        commit_msg = f"result:xp:{xp_name}"
        if stdout:
            commit_msg += f"\n\n{stdout}"

        run_command(["git", "commit", "-m", commit_msg])
    else:
        error_exit(f"Results directory not found: {results_dir}")

    print("\n✓ Experiment run complete!")


# ============================================================================
# Command: update
# ============================================================================


def cmd_rebase_helper(args: argparse.Namespace) -> None:
    """Hidden helper for interactive rebase to drop result commits."""
    todo_file = Path(args.todo_file)
    xp_name = args.xp_name

    if not todo_file.exists():
        error_exit(f"Rebase todo file not found: {todo_file}")

    with open(todo_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    # Match "pick <sha> # result:xp:<xp_name>"
    pattern = re.compile(rf"^pick\s+([a-f0-9]+)\s+#\s+result:xp:{xp_name}")

    for line in lines:
        if pattern.match(line):
            new_lines.append(line.replace("pick", "drop", 1))
        else:
            new_lines.append(line)

    with open(todo_file, "w") as f:
        f.writelines(new_lines)


def cmd_update(args: argparse.Namespace) -> None:
    """Update experiment branch by rebasing on main and removing result commits."""
    current_branch = get_current_branch()

    if args.xp_name:
        xp_name = args.xp_name
        branch_name = f"experiment/{xp_name}"
    else:
        if not current_branch.startswith("experiment/"):
            error_exit("Not on an experiment branch and no experiment name provided.")
        branch_name = current_branch
        xp_name = extract_xp_name(branch_name)

    # Checkout branch if not already on it
    if current_branch != branch_name:
        print(f"Switching to branch: {branch_name}")
        run_command(["git", "checkout", branch_name])

    print(f"Rebasing {branch_name} on origin/main and dropping result commits...")

    env = os.environ.copy()
    script_path = Path(__file__).resolve()

    # Set sequence editor to call back into this script
    env["GIT_SEQUENCE_EDITOR"] = f'"{script_path}" _rebase {xp_name}'

    # Use interactive rebase to allow the sequence editor to modify the todo list
    exit_code, _, stderr = run_command(
        ["git", "rebase", "-i", "origin/main"], check=False, env=env
    )

    if exit_code != 0:
        print(f"Rebase failed. Please resolve conflicts manually.\n{stderr}")
        sys.exit(exit_code)

    print("\n" + "=" * 80)
    print(f"✓ Experiment branch '{branch_name}' updated and cleaned!")
    print("=" * 80)
    print()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified experiment management tool for GPUSSSP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # create command
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new experiment branch",
        description="Create a new experiment branch named experiment/<xp_name>",
    )
    create_parser.add_argument(
        "xp_name", help="Experiment name (lowercase, underscores only)"
    )
    create_parser.set_defaults(func=cmd_create)

    # add command
    add_parser = subparsers.add_parser(
        "add",
        help="Add an instrumentation commit",
        description="Add an empty instrumentation commit to the current experiment branch",
    )
    add_parser.add_argument(
        "run_targets", help="Comma-separated run targets (e.g., 'deltastep,nearfar')"
    )
    add_parser.add_argument(
        "params", help="Space-separated parameters (e.g., 'delta=900,1800 data=berlin')"
    )
    add_parser.set_defaults(func=cmd_add)

    # run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run all experiments in the current branch",
        description="Run all experiments defined by instrumentation commits in the current branch",
    )
    run_parser.set_defaults(func=cmd_run)

    # update command
    update_parser = subparsers.add_parser(
        "update",
        help="Update experiment branch and clean results",
        description="Rebase experiment branch on recent main and remove result commits",
    )
    update_parser.add_argument(
        "xp_name", nargs="?", help="Experiment name (optional if on experiment branch)"
    )
    update_parser.set_defaults(func=cmd_update)

    # hidden _rebase command
    rebase_parser = subparsers.add_parser("_rebase", help=argparse.SUPPRESS)
    rebase_parser.add_argument("xp_name")
    rebase_parser.add_argument("todo_file")
    rebase_parser.set_defaults(func=cmd_rebase_helper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
