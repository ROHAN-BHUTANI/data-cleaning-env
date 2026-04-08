from __future__ import annotations

from pathlib import Path


def get_task(data_dir: Path) -> dict:
    return {
        "id": "hard",
        "description": "Resolve schema mismatch and clip outliers.",
        "dirty_path": data_dir / "hard.csv",
        "clean_path": data_dir / "hard_clean.csv",
    }
