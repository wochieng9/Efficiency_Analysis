from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def safe_rank(series: pd.Series, ascending: bool = False) -> pd.Series:
    ranks = series.rank(ascending=ascending, na_option="bottom", method="min")
    ranks = ranks.where(series.notna(), np.nan)
    return ranks.astype("Int64")

def finite_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return np.nan
    if np.nanstd(x_arr[mask]) == 0 or np.nanstd(y_arr[mask]) == 0:
        return np.nan
    return float(np.corrcoef(x_arr[mask], y_arr[mask])[0, 1])

def finite_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return np.nan
    if np.nanstd(x_arr[mask]) == 0 or np.nanstd(y_arr[mask]) == 0:
        return np.nan
    value = spearmanr(x_arr[mask], y_arr[mask]).correlation
    return float(value) if np.isfinite(value) else np.nan

def column_percentile(matrix: np.ndarray, q: float) -> np.ndarray:
    """Column-wise percentile that returns NaN for all-missing columns without warnings."""
    matrix = np.asarray(matrix, dtype=float)
    out = np.full(matrix.shape[1], np.nan)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col = col[np.isfinite(col)]
        if len(col):
            out[j] = float(np.percentile(col, q))
    return out

def column_mean(matrix: np.ndarray) -> np.ndarray:
    """Column-wise mean that returns NaN for all-missing columns without warnings."""
    matrix = np.asarray(matrix, dtype=float)
    out = np.full(matrix.shape[1], np.nan)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col = col[np.isfinite(col)]
        if len(col):
            out[j] = float(np.mean(col))
    return out

def column_std(matrix: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Column-wise standard deviation that returns NaN if too few values are available."""
    matrix = np.asarray(matrix, dtype=float)
    out = np.full(matrix.shape[1], np.nan)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col = col[np.isfinite(col)]
        if len(col) > ddof:
            out[j] = float(np.std(col, ddof=ddof))
    return out

