from __future__ import annotations

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

try:
    from .env import DataCleaningEnv
    from .models import CleaningAction, ResetRequest, ResetResponse, StateResponse, StepResponse
except ImportError:
    from env import DataCleaningEnv
    from models import CleaningAction, ResetRequest, ResetResponse, StateResponse, StepResponse


app = FastAPI(title="DataCleaningEnv", version="1.0.0")
env = DataCleaningEnv(data_dir=Path(__file__).resolve().parent.parent / "data")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "DataCleaningEnv"}


@app.get("/reset", response_model=ResetResponse)
def reset(task_id: str = "easy") -> ResetResponse:
    try:
        obs = env.reset(task_id=task_id)
        return ResetResponse(
            observation=obs,
            reward={"value": 0.0, "reason": "Environment reset"},
            done=False,
            info={"task_id": task_id},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset", response_model=ResetResponse)
def reset_post(request: ResetRequest) -> ResetResponse:
    return reset(task_id=request.task_id)


@app.post("/step", response_model=StepResponse)
def step(action: CleaningAction) -> StepResponse:
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    try:
        return StateResponse(task_id=env.current_task_id, done=env.done, observation=env.state())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/state", response_model=StateResponse)
def state_post() -> StateResponse:
    return state()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
