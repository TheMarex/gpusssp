import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from .errors import error_exit
from .gitops import run_command
from .paths import get_cache_path, get_results_dir


@dataclass
class ExperimentConfig:
    name: str
    commit_sha: str
    run_targets: List[str]
    params: Dict[str, Any]


def build_commit_message(
    xp_name: str, run_targets: List[str], params: Dict[str, Any]
) -> str:
    parts = [f"xp:{xp_name}"]

    for target in run_targets:
        parts.append(f"run:{target}")

    for key, value in params.items():
        parts.append(f"param:{key}={value}")

    return " ".join(parts)


def parse_commit_message(message: str, xp_name: str) -> Optional[ExperimentConfig]:
    expected_prefix = f"xp:{xp_name}"
    if not message.startswith(expected_prefix):
        return None

    run_targets: List[str] = []
    params: Dict[str, Any] = {}

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
        name=xp_name,
        commit_sha="",
        run_targets=run_targets,
        params=params,
    )


def build_experiment_binary(target: str, build_dir: Path) -> None:
    click.echo(f"Building experiment_{target}...")
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
    return [base_cmd + [xp_name]]


def run_experiment(
    target: str,
    config: ExperimentConfig,
    workspace_root: Path,
    build_dir: Path,
) -> None:
    cache_path = get_cache_path(workspace_root, config.params["data"])
    xp_base_path = workspace_root / "experiments" / "results"

    env = os.environ.copy()
    if "gpu" in config.params:
        env["GPUSSSP_GPU"] = str(config.params["gpu"])

    cmds = format_cmd(
        build_dir,
        target,
        cache_path,
        xp_base_path,
        config.name,
        config.params,
    )

    for cmd in cmds:
        cmd_str = " ".join(cmd)
        click.echo(f"Running '{cmd_str}'...")
        run_command(cmd, cwd=build_dir, env=env, capture=False)


def ensure_release_build(build_dir: Path, workspace_root: Path) -> None:
    click.echo("Configuring build for Release mode...")
    run_command(
        ["cmake", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=workspace_root,
    )


def collect_experiment_configs(
    commits: List[tuple[str, str]], xp_name: str
) -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for sha, message in commits:
        config = parse_commit_message(message, xp_name)
        if config:
            config.commit_sha = sha
            configs.append(config)
    return configs


def stage_and_commit_results(workspace_root: Path, xp_name: str, summary: str) -> None:
    results_dir = get_results_dir(workspace_root, xp_name)
    if not results_dir.exists():
        error_exit(f"Results directory not found: {results_dir}")

    click.echo(f"\nCommitting results from {results_dir}...")
    run_command(["git", "add", "-f", str(results_dir)])

    commit_msg = f"result:xp:{xp_name}"
    if summary:
        commit_msg += f"\n\n{summary}"

    run_command(["git", "commit", "-m", commit_msg])
