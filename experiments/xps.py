#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["click", "pandas", "matplotlib", "seaborn", "numpy"]
# ///
"""Unified experiment management tool for GPUSSSP."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.xps.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
