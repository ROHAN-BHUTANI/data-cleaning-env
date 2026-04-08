from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GradeResult:
    column_accuracy: float
    f1_score: float


class DataFrameGrader:
    _MISSING = "<MISSING>"
    _NA = "<NA>"

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        normalized = series.copy()
        normalized = normalized.astype("string").fillna(DataFrameGrader._NA)
        return normalized

    @staticmethod
    def _align_column(pred: pd.DataFrame, truth: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
        pred_series = (
            DataFrameGrader._normalize_series(pred[col])
            if col in pred.columns
            else pd.Series(dtype="string")
        )
        truth_series = DataFrameGrader._normalize_series(truth[col])

        max_rows = max(len(pred_series), len(truth_series))
        pred_series = pred_series.reindex(range(max_rows), fill_value=DataFrameGrader._MISSING)
        truth_series = truth_series.reindex(range(max_rows), fill_value=DataFrameGrader._MISSING)
        return pred_series, truth_series

    @staticmethod
    def column_wise_accuracy(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
        if truth.empty:
            return 0.0
        if pred.empty:
            return 0.0

        per_column_scores = []
        for col in truth.columns:
            pred_series, truth_series = DataFrameGrader._align_column(pred, truth, col)
            per_column_scores.append(float((pred_series == truth_series).mean()))

        schema_penalty = len(set(pred.columns).intersection(set(truth.columns))) / max(len(truth.columns), 1)
        score = (sum(per_column_scores) / len(per_column_scores)) * schema_penalty
        return max(0.0, min(1.0, score))

    @staticmethod
    def f1_score(pred: pd.DataFrame, truth: pd.DataFrame) -> float:
        if truth.empty:
            return 0.0
        if pred.empty:
            return 0.0

        tp = 0
        total_pred = 0
        total_truth = len(truth.columns) * len(truth)

        for col in truth.columns:
            pred_series, truth_series = DataFrameGrader._align_column(pred, truth, col)
            matches = pred_series == truth_series
            tp += int(matches.sum())
            total_pred += len(pred_series)

        precision = tp / max(total_pred, 1)
        recall = tp / max(total_truth, 1)
        if precision + recall == 0:
            return 0.0

        schema_penalty = len(set(pred.columns).intersection(set(truth.columns))) / max(len(truth.columns), 1)
        f1 = (2 * precision * recall) / (precision + recall)
        f1 *= schema_penalty
        if pred.empty:
            f1 *= 0.1
        return max(0.0, min(1.0, f1))

    @staticmethod
    def grade(pred: pd.DataFrame, truth: pd.DataFrame) -> GradeResult:
        return GradeResult(
            column_accuracy=DataFrameGrader.column_wise_accuracy(pred, truth),
            f1_score=DataFrameGrader.f1_score(pred, truth),
        )
