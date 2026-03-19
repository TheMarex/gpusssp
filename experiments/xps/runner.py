import itertools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

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
    metrics: List[str] = field(default_factory=lambda: ["time"])


def build_commit_message(
    xp_name: str, run_targets: List[str], params: Dict[str, Any], metrics: List[str]
) -> str:
    parts = [f"xp:{xp_name}"]

    for target in run_targets:
        parts.append(f"run:{target}")

    for key, value in params.items():
        parts.append(f"param:{key}={value}")

    for metric in metrics:
        parts.append(f"metric:{metric}")

    return " ".join(parts)


def parse_commit_message(message: str, xp_name: str) -> List[ExperimentConfig]:
    expected_prefix = f"xp:{xp_name}"
    if not message.startswith(expected_prefix):
        return []

    run_targets: List[str] = []
    params: Dict[str, List[Any]] = {}
    metrics: List[str] = []

    for match in re.finditer(r"run:(\w+)", message):
        run_targets.append(match.group(1))

    for match in re.finditer(r"param:(\w+)=([\w,]+)", message):
        param_name = match.group(1)
        param_value = match.group(2)

        if param_name in ("delta", "batch_size", "gpu"):
            params[param_name] = [int(v) for v in param_value.split(",")]
        else:
            params[param_name] = param_value.split(",")

    for match in re.finditer(r"metric:(\w+)", message):
        metrics.append(match.group(1))

    if not run_targets:
        error_exit(f"No run targets specified in commit message: {message}")
    if "data" not in params:
        error_exit(f"No param:data specified in commit message: {message}")

    if not metrics:
        metrics = ["time"]

    validate_metrics_in_commit(metrics, message)

    param_keys = list(params.keys())
    param_value_lists = [params[k] for k in param_keys]

    configs = []
    for combination in itertools.product(*param_value_lists):
        config_params = dict(zip(param_keys, combination))
        configs.append(
            ExperimentConfig(
                name=xp_name,
                commit_sha="",
                run_targets=run_targets.copy(),
                params=config_params,
                metrics=metrics.copy(),
            )
        )

    return configs


def validate_metrics_in_commit(metrics: List[str], message: str) -> None:
    if "time" in metrics and len(metrics) > 1:
        error_exit(
            f"Cannot mix 'time' metric with other metrics in commit: {message}\n"
            "'time' requires ENABLE_STATISTICS=OFF, other metrics require ENABLE_STATISTICS=ON"
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
    metrics: List[str],
) -> List[str]:
    binary = build_dir / f"experiment_{target}"
    cmd = [str(binary), str(cache_path), str(xp_base_path), "-n", xp_name]

    cmd.extend(["--metrics", ",".join(metrics)])

    if target == "dial" and "range" in params:
        cmd.extend(["--range", str(params["range"])])

    if target in ["deltastep", "nearfar"]:
        if "delta" in params:
            cmd.extend(["--delta", str(params["delta"])])
        if "batch_size" in params:
            cmd.extend(["--batch-size", str(params["batch_size"])])

    return cmd


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
        env["GPUSSSP_DEVICE"] = str(config.params["gpu"])

    cmd = format_cmd(
        build_dir,
        target,
        cache_path,
        xp_base_path,
        config.name,
        config.params,
        config.metrics,
    )

    cmd_str = " ".join(cmd)
    click.echo(f"Running '{cmd_str}'...")
    run_command(cmd, cwd=build_dir, env=env, capture=False)


def ensure_release_build(
    build_dir: Path, workspace_root: Path, metrics: List[str]
) -> None:
    enable_statistics = "time" not in metrics
    click.echo(
        f"Configuring build for Release mode (ENABLE_STATISTICS={'ON' if enable_statistics else 'OFF'})..."
    )
    run_command(
        [
            "cmake",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DENABLE_STATISTICS={'ON' if enable_statistics else 'OFF'}",
        ],
        cwd=workspace_root,
    )


def collect_experiment_configs(
    commits: List[tuple[str, str]], xp_name: str
) -> List[ExperimentConfig]:
    configs: List[ExperimentConfig] = []
    for sha, message in commits:
        parsed_configs = parse_commit_message(message, xp_name)
        for config in parsed_configs:
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
