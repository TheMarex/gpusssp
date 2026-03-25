import re
from typing import Dict, List

from .errors import error_exit

VALID_RUN_TARGETS = {
    "dijkstra",
    "deltastep",
    "bellmanford",
    "nearfar",
    "dial",
    "throughput",
}
VALID_METRICS = {"time", "edges_relaxed"}


def validate_xp_name(xp_name: str) -> None:
    if not re.match(r"^[a-z0-9_]+$", xp_name):
        error_exit(
            f"Invalid experiment name '{xp_name}'. Use only lowercase letters, numbers, and underscores."
        )


def validate_run_targets(targets: list[str]) -> None:
    for target in targets:
        if target not in VALID_RUN_TARGETS:
            error_exit(
                f"Invalid run target '{target}'. Valid targets: {', '.join(sorted(VALID_RUN_TARGETS))}"
            )


def parse_params(param_str: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for part in param_str.split():
        if "=" not in part:
            error_exit(f"Invalid param format '{part}'. Expected key=value")
        key, value = part.split("=", 1)
        params[key] = value
    return params


def validate_params(params: Dict[str, str]) -> None:
    if "data" not in params:
        error_exit("Missing required parameter: data (e.g., data=berlin)")


def parse_metrics(metrics_str: str) -> List[str]:
    if not metrics_str:
        return ["time"]
    metrics = [m.strip() for m in metrics_str.split(",")]
    return metrics


def validate_metrics(metrics: List[str]) -> None:
    for metric in metrics:
        if metric not in VALID_METRICS:
            error_exit(
                f"Invalid metric '{metric}'. Valid metrics: {', '.join(sorted(VALID_METRICS))}"
            )

    if "time" in metrics and len(metrics) > 1:
        error_exit(
            "Cannot mix 'time' metric with other metrics.\n"
            "'time' requires ENABLE_STATISTICS=OFF, other metrics require ENABLE_STATISTICS=ON.\n"
            "Use either: metric=time OR metric=edges_relaxed"
        )
