from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List

from openai import OpenAI


def _http_post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _llm_suggest_action(client: OpenAI, model_name: str, state: Dict[str, Any]) -> Dict[str, Any] | None:
    prompt = (
        "You control a data-cleaning environment. Return exactly one JSON object with keys "
        "'operation', 'column', and 'params'. Allowed actions: drop_nulls, fill_nulls, drop_duplicates, "
        "cast_column, normalize, clip_outliers, submit.\n"
        f"State: {json.dumps(state)}"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = completion.choices[0].message.content or ""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        parsed = json.loads(text[start : end + 1])
        if "operation" in parsed:
            parsed.setdefault("params", {})
            return parsed
    except Exception:
        return None
    return None


def _heuristic_plan(task_id: str) -> List[Dict[str, Any]]:
    if task_id == "easy":
        return [
            {"operation": "fill_nulls", "column": "age", "params": {"value": 28}},
            {"operation": "fill_nulls", "column": "income", "params": {"value": 57750}},
            {"operation": "cast_column", "column": "age", "params": {"dtype": "int"}},
            {"operation": "submit", "params": {}},
        ]
    if task_id == "medium":
        return [
            {"operation": "drop_duplicates", "params": {}},
            {"operation": "normalize", "column": "purchase_amount", "params": {"method": "minmax"}},
            {"operation": "submit", "params": {}},
        ]
    return [
        {"operation": "cast_column", "column": "user_id", "params": {"dtype": "int"}},
        {"operation": "clip_outliers", "column": "score", "params": {"lower": 5, "upper": 30}},
        {"operation": "submit", "params": {}},
    ]


def main() -> int:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN", "")
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    task_id = os.getenv("TASK_ID", "easy")

    client = OpenAI(base_url=api_base_url, api_key=hf_token or "dummy-token")

    print(f"[START] task={task_id} env_base_url={env_base_url} model={model_name}")

    try:
        reset_resp = _http_post(f"{env_base_url}/reset", {"task_id": task_id})
    except urllib.error.URLError as exc:
        print(f"[END] status=error error=reset_failed detail={exc}")
        return 1

    step_idx = 0
    done = bool(reset_resp.get("done", False))
    last_reward = float(reset_resp.get("reward", {}).get("value", 0.0))
    state = reset_resp
    plan = _heuristic_plan(task_id)

    while not done and step_idx < 20:
        llm_action = _llm_suggest_action(client=client, model_name=model_name, state=state)
        action = llm_action or plan[min(step_idx, len(plan) - 1)]

        try:
            step_resp = _http_post(f"{env_base_url}/step", action)
        except urllib.error.URLError as exc:
            print(f"[END] status=error error=step_failed detail={exc}")
            return 1

        step_idx += 1
        done = bool(step_resp.get("done", False))
        last_reward = float(step_resp.get("reward", {}).get("value", 0.0))
        reason = step_resp.get("reward", {}).get("reason", "")
        print(f"[STEP] n={step_idx} action={action['operation']} reward={last_reward:.6f} done={done} reason={reason}")
        state = step_resp

        if action["operation"] == "submit":
            done = True

    print(f"[END] status=ok steps={step_idx} final_reward={last_reward:.6f} done={done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
