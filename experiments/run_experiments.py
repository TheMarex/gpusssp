#!/usr/bin/env python3
"""
Automated experiment runner for GPUSSSP.

This script runs experiments based on commit messages in an experiment branch.
Branch must be named: experiment/{xp_name}
Commit messages must start with: xp:{xp_name} followed by parameters.
Example commit message:
  xp:{xp_name} run:dijkstra run:nearfar param:delta=900,1800,3600 param:data=berlin param:gpu=0
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ExperimentConfig:
    commit_sha: str
    run_targets: List[str]
    params: Dict[str, Any]


def error_exit(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def run_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> str:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_exit(f"Command failed: {' '.join(cmd)}\nOutput: {e.stderr}")


def get_current_branch() -> str:
    return run_command(["git", "branch", "--show-current"])


def extract_xp_name(branch_name: str) -> str:
    match = re.match(r"^experiment/(.+)$", branch_name)
    if not match:
        error_exit(
            f"Branch '{branch_name}' does not match pattern 'experiment/{{xp_name}}'"
        )
    return match.group(1)


def get_recent_commits(count: int = 100) -> List[tuple[str, str]]:
    output = run_command(["git", "log", f"-{count}", "--format=%H|%s"])
    commits = []
    for line in output.split("\n"):
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
        commit_sha="",
        run_targets=run_targets,
        params=params
    )


def build_experiment_binary(target: str, build_dir: Path) -> None:
    print(f"Building experiment_{target}...")
    run_command(
        ["cmake", "--build", str(build_dir), "--target", f"experiment_{target}"],
        cwd=build_dir.parent
    )


def format_cmd(
    build_dir: Path,
    target: str,
    cache_path: Path,
    xp_base_path: Path,
    xp_name: str,
    params: Dict[str, Any]
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
    target: str,
    config: ExperimentConfig,
    workspace_root: Path,
    build_dir: Path
) -> None:
    cache_path = workspace_root / "cache" / config.params["data"]
    xp_base_path = workspace_root / "experiments" / "results"
    xp_name = config.commit_sha[:9]

    env = os.environ.copy()
    if "gpu" in config.params:
        env["GPUSSSP_GPU"] = str(config.params["gpu"])

    cmds = format_cmd(build_dir, target, cache_path, xp_base_path, xp_name, config.params)
    
    for cmd in cmds:
        cmd_str = " ".join(cmd)
        print(f"Running '{cmd_str}'...")
        run_command(cmd, cwd=build_dir, env=env)


def ensure_release_build(build_dir: Path, workspace_root: Path) -> None:
    print("Configuring build for Release mode...")
    run_command(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=workspace_root
    )


def main() -> None:
    workspace_root = Path(__file__).parent.parent.resolve()
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
            print(f"Found experiment commit: {sha[:9]} - {message[:60]}...")

    if not experiment_configs:
        error_exit(f"No commits found matching 'xp:{xp_name}'")

    experiment_configs.reverse()
    print(f"\nFound {len(experiment_configs)} experiment commit(s) to process\n")

    for i, config in enumerate(experiment_configs, 1):
        print(f"\n{'='*80}")
        print(f"Processing commit {i}/{len(experiment_configs)}: {config.commit_sha[:9]}")
        print(f"{'='*80}\n")

        print(f"Checking out {config.commit_sha[:9]}...")
        run_command(["git", "checkout", config.commit_sha])

        ensure_release_build(build_dir, workspace_root)

        for target in config.run_targets:
            build_experiment_binary(target, build_dir)

        for target in config.run_targets:
            run_experiment(target, config, workspace_root, build_dir)

    print(f"\n{'='*80}")
    print("All experiments completed successfully!")
    print(f"{'='*80}\n")
    print(f"Returning to branch: {original_branch}")
    run_command(["git", "checkout", original_branch])

    results_dir = workspace_root / "experiments" / "results" / xp_name
    if results_dir.exists():
        print(f"\nCommitting results from {results_dir}...")
        run_command(["git", "add", str(results_dir)])
        run_command(["git", "commit", "-m", f"result:xp:{xp_name}"])
    else:
        error_exit(f"Results directory not found: {results_dir}")

    print("\n✓ Experiment run complete!")


if __name__ == "__main__":
    main()
