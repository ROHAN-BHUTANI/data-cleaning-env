from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

try:
    from .graders.df_grader import DataFrameGrader
    from .models import CleaningAction, Observation, Reward
    from .tasks.easy import get_task as get_easy_task
    from .tasks.hard import get_task as get_hard_task
    from .tasks.medium import get_task as get_medium_task
except ImportError:
    from graders.df_grader import DataFrameGrader
    from models import CleaningAction, Observation, Reward
    from tasks.easy import get_task as get_easy_task
    from tasks.hard import get_task as get_hard_task
    from tasks.medium import get_task as get_medium_task


class DataCleaningEnv:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._task_loaders: Dict[str, Callable[[Path], dict]] = {
            "easy": get_easy_task,
            "medium": get_medium_task,
            "hard": get_hard_task,
        }
        self.current_task_id: str | None = None
        self.current_df: pd.DataFrame | None = None
        self.ground_truth_df: pd.DataFrame | None = None
        self.step_number = 0
        self.done = False
        self.last_error = ""

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in self._task_loaders:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = self._task_loaders[task_id](self.data_dir)
        self.current_task_id = task_id
        self.current_df = pd.read_csv(task["dirty_path"])
        self.ground_truth_df = pd.read_csv(task["clean_path"])
        self.step_number = 0
        self.done = False
        self.last_error = ""
        return self._observation()

    def state(self) -> Observation:
        self._ensure_initialized()
        return self._observation()

    def step(self, action: CleaningAction) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        self._ensure_initialized()
        self.step_number += 1

        if self.done:
            reward = Reward(value=-0.05, reason="Episode already finished. Call reset().")
            self.last_error = reward.reason
            return self._observation(), reward, self.done, {"error": self.last_error}

        params = dict(action.params or {})
        if action.column is not None and "column" not in params:
            params["column"] = action.column

        try:
            if action.operation == "drop_nulls":
                self._drop_nulls(params)
                reward = Reward(value=0.01, reason="Valid action applied")
                info = {"action": action.operation}
            elif action.operation == "fill_nulls":
                self._fill_nulls(params)
                reward = Reward(value=0.01, reason="Valid action applied")
                info = {"action": action.operation}
            elif action.operation == "drop_duplicates":
                self._drop_duplicates(params)
                reward = Reward(value=0.01, reason="Valid action applied")
                info = {"action": action.operation}
            elif action.operation == "cast_column":
                self._cast_column(params)
                reward = Reward(value=0.01, reason="Valid action applied")
                info = {"action": action.operation}
            elif action.operation == "normalize":
                self._normalize(params)
                reward = Reward(value=0.01, reason="Valid action applied")
                info = {"action": action.operation}
            elif action.operation == "clip_outliers":
                self._clip_outliers(params)
                reward = Reward(value=0.01, reason="Valid action applied")
                info = {"action": action.operation}
            elif action.operation == "submit":
                grade = DataFrameGrader.grade(self.current_df, self.ground_truth_df)  # type: ignore[arg-type]
                self.done = True
                reward = Reward(value=float(grade.f1_score), reason="Submit scored against ground truth")
                info = {
                    "f1_score": grade.f1_score,
                    "column_accuracy": grade.column_accuracy,
                }
            else:
                raise ValueError(f"Unsupported action: {action.operation}")

            self.last_error = ""
            return self._observation(), reward, self.done, info
        except Exception as exc:
            self.last_error = str(exc)
            reward = Reward(value=-0.05, reason="Invalid action")
            return self._observation(), reward, self.done, {"error": self.last_error}

    def _ensure_initialized(self) -> None:
        if self.current_df is None or self.ground_truth_df is None or self.current_task_id is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

    def _observation(self) -> Observation:
        self._ensure_initialized()
        df = self.current_df  # type: ignore[assignment]
        sample = df.head(5).where(pd.notna(df), None).to_dict(orient="records")

        return Observation(
            shape=[int(df.shape[0]), int(df.shape[1])],
            columns=[str(c) for c in df.columns.tolist()],
            dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
            null_counts={str(col): int(count) for col, count in df.isna().sum().items()},
            sample_rows=sample,
            duplicate_count=int(df.duplicated().sum()),
            step_number=self.step_number,
            last_error=self.last_error,
        )

    def _drop_nulls(self, params: dict[str, Any]) -> None:
        assert self.current_df is not None
        subset = params.get("subset")
        if subset is not None and not isinstance(subset, list):
            raise ValueError("'subset' must be a list of column names")
        self.current_df = self.current_df.dropna(subset=subset).reset_index(drop=True)

    def _fill_nulls(self, params: dict[str, Any]) -> None:
        assert self.current_df is not None
        column = params.get("column")
        value = params.get("value")

        if column is None:
            if value is None:
                raise ValueError("When 'column' is omitted, provide a scalar 'value'")
            self.current_df = self.current_df.fillna(value)
            return

        if column not in self.current_df.columns:
            raise ValueError(f"Column not found: {column}")
        if value is None:
            raise ValueError("'value' is required for fill_nulls")
        self.current_df[column] = self.current_df[column].fillna(value)

    def _drop_duplicates(self, params: dict[str, Any]) -> None:
        assert self.current_df is not None
        subset = params.get("subset")
        keep = params.get("keep", "first")
        if subset is not None and not isinstance(subset, list):
            raise ValueError("'subset' must be a list")
        self.current_df = self.current_df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    def _cast_column(self, params: dict[str, Any]) -> None:
        assert self.current_df is not None
        column = params.get("column")
        dtype = params.get("dtype")
        if column is None or dtype is None:
            raise ValueError("'column' and 'dtype' are required")
        if column not in self.current_df.columns:
            raise ValueError(f"Column not found: {column}")

        if dtype == "int":
            self.current_df[column] = pd.to_numeric(self.current_df[column], errors="raise").astype("Int64")
            self.current_df[column] = self.current_df[column].astype("int64")
        elif dtype == "float":
            self.current_df[column] = pd.to_numeric(self.current_df[column], errors="raise").astype("float64")
        elif dtype == "str":
            self.current_df[column] = self.current_df[column].astype(str)
        elif dtype == "datetime":
            self.current_df[column] = pd.to_datetime(self.current_df[column], errors="raise")
        elif dtype == "bool":
            self.current_df[column] = self.current_df[column].astype(bool)
        else:
            raise ValueError("Unsupported dtype. Use one of: int, float, str, datetime, bool")

    def _normalize(self, params: dict[str, Any]) -> None:
        assert self.current_df is not None
        column = params.get("column")
        method = params.get("method", "minmax")
        if column is None:
            raise ValueError("'column' is required")
        if column not in self.current_df.columns:
            raise ValueError(f"Column not found: {column}")

        series = pd.to_numeric(self.current_df[column], errors="raise")

        if method == "minmax":
            min_v = series.min()
            max_v = series.max()
            if max_v == min_v:
                self.current_df[column] = 0.0
            else:
                self.current_df[column] = (series - min_v) / (max_v - min_v)
        elif method == "zscore":
            mean_v = series.mean()
            std_v = series.std(ddof=0)
            if std_v == 0:
                self.current_df[column] = 0.0
            else:
                self.current_df[column] = (series - mean_v) / std_v
        else:
            raise ValueError("Unsupported normalization method. Use minmax or zscore")

    def _clip_outliers(self, params: dict[str, Any]) -> None:
        assert self.current_df is not None
        column = params.get("column")
        if column is None:
            raise ValueError("'column' is required")
        if column not in self.current_df.columns:
            raise ValueError(f"Column not found: {column}")

        series = pd.to_numeric(self.current_df[column], errors="raise")

        lower = params.get("lower")
        upper = params.get("upper")
        if lower is None or upper is None:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

        self.current_df[column] = series.clip(lower=float(lower), upper=float(upper))
