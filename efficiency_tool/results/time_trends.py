from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from efficiency_tool.sfa.core import COST_EFFICIENCY_STATA
except Exception:  # pragma: no cover - defensive fallback for isolated imports
    COST_EFFICIENCY_STATA = "Stata/FRONTIER cost ratio E[exp(u)|eps]"


DEFAULT_TIME_SCORE_LABELS: Dict[str, str] = {
    "DEA_Efficiency": "DEA efficiency",
    "SFA_Efficiency": "SFA efficiency",
    "Scale_Efficiency": "Scale efficiency",
    "DEA_SuperEfficiency": "DEA super-efficiency",
    "CCR_Cross_Efficiency_Mean": "CCR cross-efficiency",
}


def _empty_time_trend_result(reason: str = "") -> Dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "time_col": None,
        "unit_col": None,
        "score_columns": [],
        "period_labels": [],
        "period_summary": pd.DataFrame(),
        "unit_change": pd.DataFrame(),
        "score_long": pd.DataFrame(),
        "change_summary": pd.DataFrame(),
    }


def _format_time_value(value: object) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return "<missing>"
        if ts == ts.normalize():
            return ts.strftime("%Y-%m-%d")
        return ts.isoformat()
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _ordered_time_map(values: pd.Series) -> Tuple[Dict[str, int], List[str]]:
    raw = pd.Series(values.dropna().tolist())
    if raw.empty:
        return {}, []

    labels = raw.map(_format_time_value)
    unique = pd.DataFrame({"Time_Period": labels, "Raw_Time_Value": raw}).drop_duplicates("Time_Period")

    numeric = pd.to_numeric(unique["Raw_Time_Value"], errors="coerce")
    if numeric.notna().all():
        unique["_sort_key"] = numeric.astype(float)
        unique = unique.sort_values(["_sort_key", "Time_Period"], kind="mergesort")
    else:
        datetimes = pd.to_datetime(unique["Raw_Time_Value"], errors="coerce")
        if datetimes.notna().all():
            unique["_sort_key"] = datetimes.astype("int64")
            unique = unique.sort_values(["_sort_key", "Time_Period"], kind="mergesort")
        else:
            unique["_sort_key"] = unique["Time_Period"].astype(str)
            unique = unique.sort_values(["_sort_key"], kind="mergesort")

    ordered_labels = unique["Time_Period"].astype(str).tolist()
    return {label: idx for idx, label in enumerate(ordered_labels)}, ordered_labels


def _available_score_columns(results_df: pd.DataFrame, score_cols: Optional[Sequence[str]] = None) -> List[str]:
    candidates = list(score_cols) if score_cols is not None else list(DEFAULT_TIME_SCORE_LABELS)
    available: List[str] = []
    for col in candidates:
        if col not in results_df.columns:
            continue
        values = pd.to_numeric(results_df[col], errors="coerce")
        if values.notna().any():
            available.append(col)
    return available


def _infer_lower_is_better_columns(results_df: pd.DataFrame) -> Set[str]:
    lower_is_better: Set[str] = set()
    if "SFA_Efficiency" in results_df.columns and "SFA_Cost_Efficiency_Convention" in results_df.columns:
        convention_values = results_df["SFA_Cost_Efficiency_Convention"].dropna().astype(str).unique().tolist()
        if any(value == COST_EFFICIENCY_STATA for value in convention_values):
            lower_is_better.add("SFA_Efficiency")
    return lower_is_better


def build_time_trend_tables(
    results_df: pd.DataFrame,
    unit_col: Optional[str],
    time_col: Optional[str],
    score_cols: Optional[Sequence[str]] = None,
    lower_is_better_cols: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Build period and unit-level first-to-last efficiency trend tables.

    The function compares the same base unit across ordered time periods. It does
    not estimate a Malmquist productivity index or a panel SFA model; it summarizes
    changes in the scores already computed by the app.
    """
    if time_col is None or time_col not in results_df.columns:
        return _empty_time_trend_result("No selected time column is available in the results table.")

    time_map, ordered_periods = _ordered_time_map(results_df[time_col])
    if len(ordered_periods) < 2:
        return _empty_time_trend_result("The selected time column has fewer than two periods after cleaning.")

    score_columns = _available_score_columns(results_df, score_cols)
    if not score_columns:
        return _empty_time_trend_result("No score columns are available for time-trend analysis.")

    if unit_col is not None and unit_col in results_df.columns:
        resolved_unit_col = unit_col
        unit_values = results_df[unit_col].astype(str)
    elif "DMU_ID" in results_df.columns:
        resolved_unit_col = "DMU_ID"
        unit_values = results_df["DMU_ID"].astype(str).str.split(" | ", regex=False).str[0]
    else:
        return _empty_time_trend_result("No unit identifier is available for within-unit trend analysis.")

    keep_cols = [resolved_unit_col, time_col] + score_columns
    work = results_df.loc[:, keep_cols].copy()
    work["Trend_Unit_ID"] = unit_values.astype(str).to_numpy()
    work["Time_Period"] = work[time_col].map(_format_time_value)
    work["Time_Order"] = work["Time_Period"].map(time_map)
    work = work[work["Time_Order"].notna()].copy()
    work["Time_Order"] = work["Time_Order"].astype(int)

    for col in score_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = (
        work.groupby(["Trend_Unit_ID", "Time_Period", "Time_Order"], dropna=False)[score_columns]
        .mean()
        .reset_index()
    )

    score_label_map = {col: DEFAULT_TIME_SCORE_LABELS.get(col, col.replace("_", " ")) for col in score_columns}
    lower_is_better = _infer_lower_is_better_columns(results_df)
    if lower_is_better_cols is not None:
        lower_is_better.update(str(col) for col in lower_is_better_cols)

    score_long = grouped.melt(
        id_vars=["Trend_Unit_ID", "Time_Period", "Time_Order"],
        value_vars=score_columns,
        var_name="Score",
        value_name="Score_Value",
    )
    score_long["Score_Label"] = score_long["Score"].map(score_label_map)
    score_long["Higher_Is_Better"] = ~score_long["Score"].isin(lower_is_better)
    score_long = score_long[np.isfinite(score_long["Score_Value"].to_numpy(dtype=float))].copy()

    if score_long.empty:
        return _empty_time_trend_result("Score columns are present, but all time-trend score values are missing.")

    period_summary = (
        score_long.groupby(["Time_Order", "Time_Period", "Score", "Score_Label", "Higher_Is_Better"], dropna=False)
        .agg(
            Units_With_Score=("Score_Value", "count"),
            Mean_Score=("Score_Value", "mean"),
            Median_Score=("Score_Value", "median"),
            Std_Score=("Score_Value", "std"),
            Min_Score=("Score_Value", "min"),
            Max_Score=("Score_Value", "max"),
        )
        .reset_index()
        .sort_values(["Score_Label", "Time_Order"], kind="mergesort")
    )

    frontier_rows: List[Dict[str, object]] = []
    for keys, sub in score_long.groupby(["Time_Order", "Time_Period", "Score", "Score_Label", "Higher_Is_Better"], dropna=False):
        time_order, period, score, label, higher = keys
        vals = sub["Score_Value"].to_numpy(dtype=float)
        if len(vals) == 0:
            share = np.nan
        elif bool(higher):
            share = float(np.mean(vals >= 0.999))
        else:
            share = float(np.mean(vals <= 1.000001))
        frontier_rows.append(
            {
                "Time_Order": time_order,
                "Time_Period": period,
                "Score": score,
                "Score_Label": label,
                "Higher_Is_Better": higher,
                "Share_At_Best_Bound": share,
            }
        )
    frontier_df = pd.DataFrame(frontier_rows)
    if not frontier_df.empty:
        period_summary = period_summary.merge(
            frontier_df,
            on=["Time_Order", "Time_Period", "Score", "Score_Label", "Higher_Is_Better"],
            how="left",
        )

    change_rows: List[Dict[str, object]] = []
    for (unit, score), sub in score_long.groupby(["Trend_Unit_ID", "Score"], dropna=False):
        sub = sub.sort_values(["Time_Order", "Time_Period"], kind="mergesort")
        if len(sub) < 2:
            continue
        first = sub.iloc[0]
        last = sub.iloc[-1]
        first_score = float(first["Score_Value"])
        last_score = float(last["Score_Value"])
        raw_change = last_score - first_score
        higher = bool(first["Higher_Is_Better"])
        direction = 1.0 if higher else -1.0
        improvement = direction * raw_change
        if np.isfinite(first_score) and abs(first_score) > 1e-12:
            raw_change_pct = 100.0 * raw_change / abs(first_score)
            improvement_pct = 100.0 * improvement / abs(first_score)
        else:
            raw_change_pct = np.nan
            improvement_pct = np.nan
        change_rows.append(
            {
                "Trend_Unit_ID": unit,
                "Score": score,
                "Score_Label": score_label_map.get(str(score), str(score)),
                "Higher_Is_Better": higher,
                "First_Period": str(first["Time_Period"]),
                "Last_Period": str(last["Time_Period"]),
                "Periods_Observed": int(sub["Time_Period"].nunique()),
                "First_Score": first_score,
                "Last_Score": last_score,
                "Raw_Change": raw_change,
                "Raw_Change_Percent_of_First": raw_change_pct,
                "Improvement": improvement,
                "Improvement_Percent_of_First": improvement_pct,
                "Improved": bool(improvement > 1e-8),
                "Declined": bool(improvement < -1e-8),
                "Unchanged": bool(abs(improvement) <= 1e-8),
            }
        )

    unit_change = pd.DataFrame(change_rows)
    if not unit_change.empty:
        unit_change = unit_change.sort_values(["Score_Label", "Improvement"], ascending=[True, False], kind="mergesort")
        change_summary = (
            unit_change.groupby(["Score", "Score_Label", "Higher_Is_Better"], dropna=False)
            .agg(
                Units_Compared=("Trend_Unit_ID", "count"),
                Average_First_Score=("First_Score", "mean"),
                Average_Last_Score=("Last_Score", "mean"),
                Average_Raw_Change=("Raw_Change", "mean"),
                Average_Improvement=("Improvement", "mean"),
                Median_Improvement=("Improvement", "median"),
                Share_Improved=("Improved", "mean"),
                Share_Declined=("Declined", "mean"),
            )
            .reset_index()
            .sort_values("Score_Label", kind="mergesort")
        )
    else:
        change_summary = pd.DataFrame()

    return {
        "available": True,
        "reason": "",
        "time_col": time_col,
        "unit_col": resolved_unit_col,
        "score_columns": score_columns,
        "period_labels": ordered_periods,
        "period_summary": period_summary.reset_index(drop=True),
        "unit_change": unit_change.reset_index(drop=True),
        "score_long": score_long.sort_values(["Score_Label", "Trend_Unit_ID", "Time_Order"], kind="mergesort").reset_index(drop=True),
        "change_summary": change_summary.reset_index(drop=True),
    }
