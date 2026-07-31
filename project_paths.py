"""Portable paths for repository-bundled and deployment-specific resources."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(os.environ.get("WEBAPP_ROOT", Path(__file__).resolve().parent))


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def state_pattern_dir() -> Path:
    return Path(os.environ.get("STATE_PATTERN_DIR", repo_path("State_pattern_files")))


def postprocess_dir() -> Path:
    return Path(os.environ.get("POSTPROCESS_DIR", repo_path("postprocess")))


def upload_dir() -> Path:
    return Path(os.environ.get("UPLOAD_DIR", repo_path("uploaded_files")))


def static_plots_dir() -> Path:
    return Path(os.environ.get("STATIC_PLOTS_DIR", repo_path("static", "plots")))


def ensure_on_python_path(directory: Path) -> None:
    path = str(directory)
    if path not in sys.path:
        sys.path.insert(0, path)
