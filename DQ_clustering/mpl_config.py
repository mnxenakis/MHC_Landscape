"""Matplotlib runtime setup for non-interactive release workflows."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib_cache() -> None:
    """Use a writable Matplotlib cache directory unless the user set one."""
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_dir = Path(tempfile.gettempdir()) / f"matplotlib_{os.getuid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
