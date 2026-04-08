from __future__ import annotations

from pathlib import Path


def get_task(data_dir: Path) -> dict:
    return {
        "id": "medium",
        "description": "Remove duplicates and normalize numeric columns.",
        "dirty_path": data_dir / "medium.csv",
        "clean_path": data_dir / "medium_clean.csv",
    }
