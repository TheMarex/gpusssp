from pathlib import Path


def get_workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_build_dir(workspace_root: Path) -> Path:
    return workspace_root / "build"


def get_results_dir(workspace_root: Path, xp_name: str) -> Path:
    return workspace_root / "experiments" / "results" / xp_name


def get_cache_path(workspace_root: Path, dataset_name: str) -> Path:
    return workspace_root / "cache" / dataset_name


def get_launcher_path() -> Path:
    """Return the legacy script path used by git sequence editor hooks."""

    return get_workspace_root() / "experiments" / "xps.py"
