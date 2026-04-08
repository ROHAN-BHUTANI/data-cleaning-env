from __future__ import annotations

import sys
from pathlib import Path

import requests
import yaml


REQUIRED_ACTIONS = {
    "drop_nulls",
    "fill_nulls",
    "drop_duplicates",
    "cast_column",
    "normalize",
    "clip_outliers",
    "submit",
}

REQUIRED_OBS_FIELDS = {
    "shape",
    "columns",
    "dtypes",
    "null_counts",
    "sample_rows",
    "duplicate_count",
    "step_number",
    "last_error",
}


def fail(message: str) -> int:
    print(f"openenv validate failed: {message}")
    return 1


def _assert_yaml_contract(doc: dict) -> None:
    if doc.get("name") != "DataCleaningEnv":
        raise ValueError("name must be DataCleaningEnv")

    server = doc.get("server", {})
    endpoints = server.get("endpoints", {})
    if endpoints.get("step", {}).get("path") != "/step" or endpoints.get("step", {}).get("method") != "POST":
        raise ValueError("step endpoint must be POST /step")
    if endpoints.get("reset", {}).get("path") != "/reset" or endpoints.get("reset", {}).get("method") != "GET":
        raise ValueError("reset endpoint must be GET /reset")
    if endpoints.get("state", {}).get("path") != "/state" or endpoints.get("state", {}).get("method") != "GET":
        raise ValueError("state endpoint must be GET /state")

    action_space = set(doc.get("action_space", []))
    if action_space != REQUIRED_ACTIONS:
        raise ValueError("action_space does not match required actions")

    obs_fields = set(doc.get("observation_fields", []))
    if obs_fields != REQUIRED_OBS_FIELDS:
        raise ValueError("observation_fields do not match required observation contract")


def _assert_live_api(base_url: str) -> None:
    reset = requests.get(f"{base_url}/reset", timeout=10)
    reset.raise_for_status()
    reset_json = reset.json()

    for key in ("observation", "reward", "done"):
        if key not in reset_json:
            raise ValueError(f"/reset missing key: {key}")

    step = requests.post(
        f"{base_url}/step",
        json={"operation": "drop_nulls"},
        timeout=10,
    )
    step.raise_for_status()
    step_json = step.json()

    for key in ("observation", "reward", "done"):
        if key not in step_json:
            raise ValueError(f"/step missing key: {key}")

    obs = step_json["observation"]
    for field in REQUIRED_OBS_FIELDS:
        if field not in obs:
            raise ValueError(f"observation missing field: {field}")


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = root / "openenv.yaml"
    if not manifest.exists():
        return fail("openenv.yaml not found")

    with manifest.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    try:
        _assert_yaml_contract(doc)
    except Exception as exc:
        return fail(str(exc))

    try:
        _assert_live_api("http://localhost:8000")
    except Exception as exc:
        return fail(f"live API check failed: {exc}")

    print("openenv validate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
