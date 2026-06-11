from __future__ import annotations

import math
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from efficiency_tool.config import TOL
from efficiency_tool.dea.bootstrap import solve_dea_against_reference


def _empty_malmquist_result(reason: str) -> Dict[str, object]:
    return {
        "available": False,
        "reason": reason,
        "summary": pd.DataFrame(),
        "pairwise": pd.DataFrame(),
        "chain": pd.DataFrame(),
        "period_pairs": pd.DataFrame(),
        "panel": pd.DataFrame(),
        "period_labels": [],
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

    ordered = unique["Time_Period"].astype(str).tolist()
    return {label: idx for idx, label in enumerate(ordered)}, ordered


def _safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(float(denom)) <= TOL:
        return np.nan
    return float(numer) / float(denom)


def _safe_sqrt_product(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan
    product = float(a) * float(b)
    if product < 0:
        return np.nan
    return math.sqrt(product)


def _evaluate_distance_score(
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    orientation: str,
    returns: str,
) -> Tuple[float, bool, str]:
    """Return a Malmquist distance-style DEA score for one observation.

    For input orientation this is the Farrell input score theta. For output
    orientation this is 1 / phi, where phi is the output expansion factor.
    This keeps the MPI convention consistent: values above 1 for the final
    Malmquist index indicate productivity improvement.
    """
    result = solve_dea_against_reference(
        X_ref,
        Y_ref,
        x_eval,
        y_eval,
        orientation=orientation,
        returns=returns,
    )
    if not bool(result.get("success", False)):
        return np.nan, False, str(result.get("message", "DEA LP failed."))

    value = float(result.get("raw_score", np.nan))
    if not np.isfinite(value):
        return np.nan, False, "DEA LP returned a non-finite score."
    return value, True, str(result.get("message", ""))


def _build_panel_table(
    data: pd.DataFrame,
    unit_col: str,
    time_col: str,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
) -> pd.DataFrame:
    keep_cols = [unit_col, time_col] + list(input_cols) + list(output_cols)
    work = data.loc[:, keep_cols].copy()
    work["Trend_Unit_ID"] = work[unit_col].astype(str)
    work["Time_Period"] = work[time_col].map(_format_time_value)

    for col in list(input_cols) + list(output_cols):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    agg_map = {col: "mean" for col in list(input_cols) + list(output_cols)}
    panel = (
        work.groupby(["Trend_Unit_ID", "Time_Period"], dropna=False)
        .agg(**{col: (col, "mean") for col in list(input_cols) + list(output_cols)}, Rows_Aggregated=(unit_col, "size"))
        .reset_index()
    )
    return panel


def _reference_arrays(
    panel: pd.DataFrame,
    period_label: str,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    reference_units: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ref = panel[panel["Time_Period"] == period_label].copy()
    if reference_units is not None:
        unit_set = set(map(str, reference_units))
        ref = ref[ref["Trend_Unit_ID"].astype(str).isin(unit_set)].copy()
    ref = ref.sort_values("Trend_Unit_ID", kind="mergesort").reset_index(drop=True)
    X_ref = ref.loc[:, list(input_cols)].to_numpy(dtype=float)
    Y_ref = ref.loc[:, list(output_cols)].to_numpy(dtype=float)
    return X_ref, Y_ref, ref


def _period_rows_by_unit(panel: pd.DataFrame, period_label: str) -> Dict[str, pd.Series]:
    sub = panel[panel["Time_Period"] == period_label].copy()
    return {str(row["Trend_Unit_ID"]): row for _, row in sub.iterrows()}


def _build_chain_table(pairwise: pd.DataFrame, period_labels: Sequence[str]) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for unit, sub in pairwise.groupby("Trend_Unit_ID", dropna=False):
        sub = sub.sort_values(["Period_1_Order", "Period_0_Order"], kind="mergesort")
        first = sub.iloc[0]
        mpi_chain = 1.0
        ec_chain = 1.0
        tc_chain = 1.0
        rows.append(
            {
                "Trend_Unit_ID": unit,
                "Time_Period": str(first["Period_0"]),
                "Time_Order": int(first["Period_0_Order"]),
                "Adjacent_From_Period": "",
                "Adjacent_MPI": 1.0,
                "Adjacent_Efficiency_Change": 1.0,
                "Adjacent_Technical_Change": 1.0,
                "MPI_Chain_Index": mpi_chain,
                "Efficiency_Change_Chain_Index": ec_chain,
                "Technical_Change_Chain_Index": tc_chain,
            }
        )
        for _, row in sub.iterrows():
            mpi = float(row["Malmquist_MPI"])
            ec = float(row["Efficiency_Change_EC"])
            tc = float(row["Technical_Change_TC"])
            if np.isfinite(mpi):
                mpi_chain *= mpi
            else:
                mpi_chain = np.nan
            if np.isfinite(ec):
                ec_chain *= ec
            else:
                ec_chain = np.nan
            if np.isfinite(tc):
                tc_chain *= tc
            else:
                tc_chain = np.nan
            rows.append(
                {
                    "Trend_Unit_ID": unit,
                    "Time_Period": str(row["Period_1"]),
                    "Time_Order": int(row["Period_1_Order"]),
                    "Adjacent_From_Period": str(row["Period_0"]),
                    "Adjacent_MPI": mpi,
                    "Adjacent_Efficiency_Change": ec,
                    "Adjacent_Technical_Change": tc,
                    "MPI_Chain_Index": mpi_chain,
                    "Efficiency_Change_Chain_Index": ec_chain,
                    "Technical_Change_Chain_Index": tc_chain,
                }
            )

    chain = pd.DataFrame(rows).drop_duplicates(["Trend_Unit_ID", "Time_Period"], keep="last")
    return chain.sort_values(["Trend_Unit_ID", "Time_Order"], kind="mergesort").reset_index(drop=True)


def calculate_malmquist_indices(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    unit_col: Optional[str],
    time_col: Optional[str],
    orientation: str = "input",
    returns: str = "vrs",
    same_reference: bool = False,
) -> Dict[str, object]:
    """Calculate adjacent-period DEA Malmquist productivity indices.

    The implementation follows the pairwise DEA approach used by common DEA
    packages: for each adjacent pair of periods, estimate four distance-style DEA
    scores e00, e01, e10, and e11, then compute:

        EC  = e11 / e00
        TC  = sqrt((e10 / e11) * (e00 / e01))
        MPI = EC * TC = sqrt((e10 / e00) * (e11 / e01))

    Values above 1 indicate productivity improvement under this convention.
    """
    if unit_col is None or time_col is None:
        return _empty_malmquist_result("Malmquist requires both a DMU identifier column and a time column.")
    if unit_col not in data.columns or time_col not in data.columns:
        return _empty_malmquist_result("The selected DMU identifier or time column is not available after cleaning.")
    if not input_cols or not output_cols:
        return _empty_malmquist_result("Malmquist requires at least one DEA input and one DEA output.")

    missing_cols = [col for col in list(input_cols) + list(output_cols) if col not in data.columns]
    if missing_cols:
        return _empty_malmquist_result("Malmquist columns are missing: " + ", ".join(map(str, missing_cols)))

    time_map, period_labels = _ordered_time_map(data[time_col])
    if len(period_labels) < 2:
        return _empty_malmquist_result("The selected time column has fewer than two periods after cleaning.")

    panel = _build_panel_table(data, str(unit_col), str(time_col), input_cols, output_cols)
    panel["Time_Order"] = panel["Time_Period"].map(time_map)
    panel = panel[panel["Time_Order"].notna()].copy()
    panel["Time_Order"] = panel["Time_Order"].astype(int)

    model_cols = list(input_cols) + list(output_cols)
    finite_positive = np.isfinite(panel[model_cols].to_numpy(dtype=float)).all(axis=1) & (panel[model_cols].to_numpy(dtype=float) > 0).all(axis=1)
    panel = panel.loc[finite_positive].reset_index(drop=True)
    if panel.empty:
        return _empty_malmquist_result("No positive finite DMU-period rows are available for Malmquist.")

    pair_rows: List[Dict[str, object]] = []
    period_pair_rows: List[Dict[str, object]] = []

    orientation_norm = str(orientation).strip().lower()
    returns_norm = str(returns).strip().lower()

    for t0_order in range(len(period_labels) - 1):
        period0 = period_labels[t0_order]
        period1 = period_labels[t0_order + 1]
        rows0 = _period_rows_by_unit(panel, period0)
        rows1 = _period_rows_by_unit(panel, period1)
        common_units = sorted(set(rows0).intersection(rows1))

        if not common_units:
            period_pair_rows.append(
                {
                    "Period_0": period0,
                    "Period_1": period1,
                    "Period_0_Order": t0_order,
                    "Period_1_Order": t0_order + 1,
                    "Units_In_Period_0": len(rows0),
                    "Units_In_Period_1": len(rows1),
                    "Common_Units_Compared": 0,
                    "Reference_Units_Period_0": 0,
                    "Reference_Units_Period_1": 0,
                    "Finite_MPI_Count": 0,
                    "LP_Success_Rate": np.nan,
                }
            )
            continue

        ref_units = common_units if same_reference else None
        X_ref0, Y_ref0, ref0 = _reference_arrays(panel, period0, input_cols, output_cols, ref_units)
        X_ref1, Y_ref1, ref1 = _reference_arrays(panel, period1, input_cols, output_cols, ref_units)

        success_count = 0
        attempt_count = 0
        finite_mpi_count = 0

        for unit in common_units:
            row0 = rows0[unit]
            row1 = rows1[unit]
            x0 = row0.loc[list(input_cols)].to_numpy(dtype=float)
            y0 = row0.loc[list(output_cols)].to_numpy(dtype=float)
            x1 = row1.loc[list(input_cols)].to_numpy(dtype=float)
            y1 = row1.loc[list(output_cols)].to_numpy(dtype=float)

            e00, ok00, msg00 = _evaluate_distance_score(X_ref0, Y_ref0, x0, y0, orientation_norm, returns_norm)
            e01, ok01, msg01 = _evaluate_distance_score(X_ref1, Y_ref1, x0, y0, orientation_norm, returns_norm)
            e10, ok10, msg10 = _evaluate_distance_score(X_ref0, Y_ref0, x1, y1, orientation_norm, returns_norm)
            e11, ok11, msg11 = _evaluate_distance_score(X_ref1, Y_ref1, x1, y1, orientation_norm, returns_norm)

            statuses = [ok00, ok01, ok10, ok11]
            messages = [msg00, msg01, msg10, msg11]
            success_count += int(sum(statuses))
            attempt_count += 4

            ec = _safe_ratio(e11, e00)
            tc = _safe_sqrt_product(_safe_ratio(e10, e11), _safe_ratio(e00, e01))
            mpi = _safe_sqrt_product(_safe_ratio(e10, e00), _safe_ratio(e11, e01))
            if np.isfinite(mpi):
                finite_mpi_count += 1

            pair_rows.append(
                {
                    "Trend_Unit_ID": unit,
                    "Period_0": period0,
                    "Period_1": period1,
                    "Period_0_Order": t0_order,
                    "Period_1_Order": t0_order + 1,
                    "Malmquist_MPI": mpi,
                    "Efficiency_Change_EC": ec,
                    "Technical_Change_TC": tc,
                    "Productivity_Change_Percent": (mpi - 1.0) * 100.0 if np.isfinite(mpi) else np.nan,
                    "Efficiency_Change_Percent": (ec - 1.0) * 100.0 if np.isfinite(ec) else np.nan,
                    "Technical_Change_Percent": (tc - 1.0) * 100.0 if np.isfinite(tc) else np.nan,
                    "Productivity_Improved": bool(np.isfinite(mpi) and mpi > 1.0 + 1e-8),
                    "Productivity_Declined": bool(np.isfinite(mpi) and mpi < 1.0 - 1e-8),
                    "Productivity_Unchanged": bool(np.isfinite(mpi) and abs(mpi - 1.0) <= 1e-8),
                    "e00_Period0_On_Period0_Frontier": e00,
                    "e01_Period0_On_Period1_Frontier": e01,
                    "e10_Period1_On_Period0_Frontier": e10,
                    "e11_Period1_On_Period1_Frontier": e11,
                    "All_Distance_LPs_Succeeded": bool(all(statuses)),
                    "Distance_LP_Success_Count": int(sum(statuses)),
                    "Distance_LP_Messages": " | ".join(sorted(set(msg for msg in messages if msg))),
                    "Rows_Aggregated_Period_0": int(row0.get("Rows_Aggregated", 1)),
                    "Rows_Aggregated_Period_1": int(row1.get("Rows_Aggregated", 1)),
                    "Reference_Units_Period_0": int(len(ref0)),
                    "Reference_Units_Period_1": int(len(ref1)),
                    "Orientation": orientation_norm,
                    "Returns_to_Scale": returns_norm,
                    "Same_Reference_Units": bool(same_reference),
                }
            )

        period_pair_rows.append(
            {
                "Period_0": period0,
                "Period_1": period1,
                "Period_0_Order": t0_order,
                "Period_1_Order": t0_order + 1,
                "Units_In_Period_0": len(rows0),
                "Units_In_Period_1": len(rows1),
                "Common_Units_Compared": len(common_units),
                "Reference_Units_Period_0": int(len(ref0)),
                "Reference_Units_Period_1": int(len(ref1)),
                "Finite_MPI_Count": finite_mpi_count,
                "LP_Success_Rate": success_count / attempt_count if attempt_count else np.nan,
            }
        )

    pairwise = pd.DataFrame(pair_rows)
    period_pairs = pd.DataFrame(period_pair_rows)
    if pairwise.empty:
        return _empty_malmquist_result("No units were present in two adjacent periods, so no Malmquist indices could be calculated.")

    if pairwise["Malmquist_MPI"].notna().sum() == 0:
        reason = (
            "Malmquist LPs ran, but no finite MPI values were produced. This can happen with VRS cross-period "
            "frontiers when observations from one period are outside the other period's feasible technology."
        )
    else:
        reason = ""

    summary = (
        pairwise.groupby(["Period_0_Order", "Period_1_Order", "Period_0", "Period_1"], dropna=False)
        .agg(
            Units_Compared=("Trend_Unit_ID", "count"),
            Finite_MPI_Count=("Malmquist_MPI", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            Mean_MPI=("Malmquist_MPI", "mean"),
            Median_MPI=("Malmquist_MPI", "median"),
            Mean_Efficiency_Change=("Efficiency_Change_EC", "mean"),
            Mean_Technical_Change=("Technical_Change_TC", "mean"),
            Share_Productivity_Improved=("Productivity_Improved", "mean"),
            Share_Productivity_Declined=("Productivity_Declined", "mean"),
            LP_Success_Rate=("Distance_LP_Success_Count", lambda s: float(np.nansum(s.to_numpy(dtype=float))) / (4.0 * len(s)) if len(s) else np.nan),
        )
        .reset_index()
        .sort_values(["Period_1_Order", "Period_0_Order"], kind="mergesort")
    )

    chain = _build_chain_table(pairwise[pairwise["Malmquist_MPI"].notna()].copy(), period_labels)

    return {
        "available": bool(pairwise["Malmquist_MPI"].notna().any()),
        "reason": reason,
        "summary": summary.reset_index(drop=True),
        "pairwise": pairwise.sort_values(["Period_1_Order", "Trend_Unit_ID"], kind="mergesort").reset_index(drop=True),
        "chain": chain,
        "period_pairs": period_pairs.sort_values(["Period_1_Order", "Period_0_Order"], kind="mergesort").reset_index(drop=True),
        "panel": panel.sort_values(["Trend_Unit_ID", "Time_Order"], kind="mergesort").reset_index(drop=True),
        "period_labels": list(period_labels),
        "orientation": orientation_norm,
        "returns": returns_norm,
        "same_reference": bool(same_reference),
    }
