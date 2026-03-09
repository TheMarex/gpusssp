import re
from typing import Dict

from .errors import error_exit

VALID_RUN_TARGETS = {"dijkstra", "deltastep", "bellmanford", "nearfar", "dial"}


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
