from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import subprocess

from .errors import error_exit


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture: bool = True,
) -> Tuple[int, str, str]:
    """Run command and return (exit_code, stdout, stderr)."""

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture,
        text=True,
        check=False,
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
    import re

    match = re.match(r"^experiment/(.+)$", branch_name)
    if match:
        return match.group(1)

    error_exit(
        f"Branch '{branch_name}' does not match pattern 'experiment/{{xp_name}}'"
    )
    return ""


def get_recent_commits(count: int = 100) -> List[Tuple[str, str]]:
    _, stdout, _ = run_command(["git", "log", f"-{count}", "--format=%H|%s"])
    commits = []
    for line in stdout.split("\n"):
        if "|" in line:
            sha, message = line.split("|", 1)
            commits.append((sha, message))
    return commits
