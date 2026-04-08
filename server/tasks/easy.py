from __future__ import annotations

from pathlib import Path


def get_task(data_dir: Path) -> dict:
    return {
        "id": "easy",
        "description": "Fix null values and type issues.",
        "dirty_path": data_dir / "easy.csv",
        "clean_path": data_dir / "easy_clean.csv",
    }
