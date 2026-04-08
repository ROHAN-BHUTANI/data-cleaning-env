from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "drop_nulls",
    "fill_nulls",
    "drop_duplicates",
    "cast_column",
    "normalize",
    "clip_outliers",
    "submit",
]


class Action(BaseModel):
    action: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class CleaningAction(BaseModel):
    operation: str
    column: Optional[str] = None
    params: Optional[dict] = None


class Observation(BaseModel):
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    sample_rows: List[Dict[str, Any]]
    duplicate_count: int
    step_number: int
    last_error: str = ""


class Reward(BaseModel):
    value: float
    reason: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"] = "easy"


class ResetResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    task_id: Optional[str] = None
    done: bool
    observation: Observation
