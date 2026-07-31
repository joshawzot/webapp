"""Portable state-pattern file lookup."""

from __future__ import annotations

from pathlib import Path

from project_paths import state_pattern_dir


SPECIAL_PATTERN_NAMES = {
    "ecc_2048x32_78tables": "SPECIAL_78TABLES",
}


def get_pattern_files() -> dict[str, str]:
    """Return known pattern names mapped to absolute, portable file paths."""
    directory = state_pattern_dir()
    patterns = {
        file.stem: str(file)
        for file in directory.glob("*.npy")
    }
    if (ecc_new := directory / "ecc_new.npy").is_file():
        patterns["test_chin"] = str(ecc_new)
    return patterns


def get_pattern_file(pattern_name: str) -> str:
    """Resolve a pattern name independently of the current working directory."""
    if pattern_name in SPECIAL_PATTERN_NAMES:
        return SPECIAL_PATTERN_NAMES[pattern_name]
    if pattern_name.startswith("ecc_2048x32_combined_"):
        return "SPECIAL_ECC_SUBSET"
    return get_pattern_files().get(pattern_name, "")
