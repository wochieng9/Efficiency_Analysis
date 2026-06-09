"""
Improved Streamlit app for Data Envelopment Analysis (DEA) and
Stochastic Frontier Analysis (SFA).

Major fixes versus the original prototype:
- Correct output-oriented DEA scoring: reports 1 / phi as a 0-1 efficiency score.
- Adds DEA targets, peers, slacks, radial adjustments, solver diagnostics, CRS/VRS
  scale efficiency, and returns-to-scale diagnostics.
- Replaces the earlier pseudo cross-efficiency label with two separate measures:
  benchmark reference support and CCR multiplier cross-efficiency.
- Makes SFA explicitly single-output, adds Cobb-Douglas and Translog production
  frontiers, numerically stable half-normal ML estimation, convergence diagnostics,
  gamma, AIC, BIC, and standard-error placeholders/approximations.
- Preserves DMU identifiers and optional time, group, and environmental columns.
- Adds cached computation, safer validation, optional DEA bootstrap robustness, robust downloads, and plot cleanup.
"""

from __future__ import annotations

import io
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog, minimize
from scipy.stats import norm, spearmanr


APP_TITLE = "Advanced Stochastic Frontier & Data Envelopment Analysis"
TOL = 1e-8
PEER_TOL = 1e-6


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def unique_keep_order(items: Iterable[Optional[str]]) -> List[str]:
    """Return unique non-empty column names while preserving order."""
    seen = set()
    out: List[str] = []
    for item in items:
        if item is None or item == "":
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file from bytes."""
    suffix = file_name.lower().split(".")[-1]
    buffer = io.BytesIO(file_bytes)
    if suffix == "csv":
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


def likely_numeric_columns(df: pd.DataFrame, threshold: float = 0.80) -> List[str]:
    """Find columns that are probably usable as numeric model variables."""
    candidates: List[str] = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        share_numeric = converted.notna().mean() if len(converted) else 0
        if share_numeric >= threshold:
            candidates.append(col)
    return candidates


def add_sidebar_selectbox(label: str, columns: Sequence[str], help_text: str = "") -> Optional[str]:
    """Select one optional column from a dataframe."""
    options: List[Optional[str]] = [None] + list(columns)
    return st.sidebar.selectbox(
        label,
        options=options,
        format_func=lambda x: "None" if x is None else str(x),
        help=help_text,
    )


def clean_and_prepare_data(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    id_col: Optional[str],
    time_col: Optional[str],
    group_col: Optional[str],
    env_cols: Sequence[str],
    selected_groups: Optional[Sequence[object]] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Preserve metadata, coerce model columns to numeric, and drop invalid model rows."""
    model_cols = unique_keep_order(list(input_cols) + list(output_cols))
    metadata_cols = unique_keep_order([id_col, time_col, group_col] + list(env_cols))
    keep_cols = unique_keep_order(metadata_cols + model_cols)

    working = df.loc[:, keep_cols].copy()
    working["Original_Row"] = df.index + 1

    if group_col is not None and selected_groups:
        working = working[working[group_col].isin(selected_groups)].copy()

    for col in model_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    missing_mask = working[model_cols].isna().any(axis=1)
    nonpositive_mask = (working[model_cols] <= 0).any(axis=1)
    invalid_mask = missing_mask | nonpositive_mask

    clean = working.loc[~invalid_mask].copy()
    if id_col is not None:
        clean["DMU_ID"] = clean[id_col].astype(str)
    else:
        clean["DMU_ID"] = clean["Original_Row"].map(lambda x: f"DMU_{int(x)}")

    duplicate_ids_before_fix = int(clean["DMU_ID"].duplicated().sum())
    if duplicate_ids_before_fix > 0:
        clean["DMU_ID"] = (
            clean["DMU_ID"].astype(str)
            + " (row "
            + clean["Original_Row"].astype(int).astype(str)
            + ")"
        )

    summary: Dict[str, object] = {
        "rows_original": int(len(df)),
        "rows_after_group_filter": int(len(working)),
        "rows_removed_missing": int(missing_mask.sum()),
        "rows_removed_nonpositive": int((nonpositive_mask & ~missing_mask).sum()),
        "rows_clean": int(len(clean)),
        "duplicate_ids": duplicate_ids_before_fix,
    }
    return clean.reset_index(drop=True), summary


def validate_analysis_setup(
    df_clean: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    id_col: Optional[str],
    sfa_model: str,
) -> Tuple[List[str], List[str]]:
    """Return blocking errors and non-blocking warnings for the selected setup."""
    errors: List[str] = []
    warnings: List[str] = []

    overlap = sorted(set(input_cols).intersection(output_cols))
    if overlap:
        errors.append("Input and output columns must be disjoint: " + ", ".join(overlap))

    if not input_cols:
        errors.append("Select at least one input column.")
    if not output_cols:
        errors.append("Select at least one output column.")

    if len(df_clean) == 0:
        errors.append("No usable rows remain after cleaning missing and non-positive model values.")

    if id_col is not None and "DMU_ID" in df_clean and df_clean["DMU_ID"].duplicated().any():
        warnings.append("Duplicate DMU identifiers were found. Rankings and peer tables may be harder to interpret.")

    model_cols = list(input_cols) + list(output_cols)
    for col in model_cols:
        if col in df_clean and df_clean[col].nunique(dropna=True) <= 1:
            warnings.append(f"Column '{col}' is constant after cleaning; it may weaken DEA/SFA identification.")

    n = len(df_clean)
    m = len(input_cols)
    s = len(output_cols)
    if n > 0:
        rule_of_thumb = max(m * s, 3 * (m + s))
        if n < rule_of_thumb:
            warnings.append(
                f"DEA has {n} DMUs for {m} inputs and {s} outputs. "
                f"A common rule of thumb is at least max(m*s, 3*(m+s)) = {rule_of_thumb} DMUs."
            )

        log_values = np.log(df_clean[model_cols].astype(float))
        zscores = (log_values - log_values.mean()) / log_values.std(ddof=0).replace(0, np.nan)
        outlier_rows = zscores.abs().gt(3).any(axis=1).sum()
        if outlier_rows:
            warnings.append(
                f"{int(outlier_rows)} rows have at least one selected variable more than 3 log-standard-deviations from the mean. "
                "Check for outliers or unit-of-measure errors."
            )

    # SFA degrees-of-freedom warning.
    if n > 0 and input_cols:
        cd_params = 1 + len(input_cols)
        translog_params = cd_params + len(input_cols) + (len(input_cols) * (len(input_cols) - 1)) // 2
        p = translog_params if sfa_model == "Translog" else cd_params
        if n <= p + 3:
            warnings.append(
                f"The selected {sfa_model} SFA has {p} frontier coefficients plus variance terms for {n} rows. "
                "SFA estimates may be unstable."
            )

    return errors, warnings


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


def display_download_button(label: str, data: bytes, file_name: str, mime: str) -> None:
    """Use on_click='ignore' when supported, with a fallback for older Streamlit versions."""
    try:
        st.download_button(
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            on_click="ignore",
        )
    except TypeError:
        st.download_button(label=label, data=data, file_name=file_name, mime=mime)


# -----------------------------------------------------------------------------
# DEA: envelopment model, slacks, targets, super-efficiency, scale diagnostics
# -----------------------------------------------------------------------------


def solve_dea_envelopment(
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
    orientation: str = "output",
    returns: str = "crs",
    exclude_k: bool = False,
) -> Dict[str, object]:
    """Solve one DEA envelopment LP for DMU k."""
    n_units, n_inputs = X.shape
    n_outputs = Y.shape[1]

    ref_indices = [j for j in range(n_units) if not (exclude_k and j == k)]
    n_ref = len(ref_indices)
    full_lambdas = np.zeros(n_units)

    if n_ref == 0:
        return {
            "success": False,
            "status": -1,
            "message": "No reference DMUs available.",
            "factor": np.nan,
            "lambdas": full_lambdas,
        }

    X_ref = X[ref_indices, :]
    Y_ref = Y[ref_indices, :]

    c = np.zeros(n_ref + 1)
    A_ub_list: List[np.ndarray] = []
    b_ub_list: List[float] = []

    if orientation == "output":
        # Maximize phi subject to X lambda <= x_k and Y lambda >= phi y_k.
        # linprog minimizes, so minimize -phi.
        c[-1] = -1.0

        for i in range(n_inputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = X_ref[:, i]
            A_ub_list.append(row)
            b_ub_list.append(float(X[k, i]))

        for r in range(n_outputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = -Y_ref[:, r]
            row[-1] = Y[k, r]
            A_ub_list.append(row)
            b_ub_list.append(0.0)

    else:
        # Minimize theta subject to X lambda <= theta x_k and Y lambda >= y_k.
        c[-1] = 1.0

        for i in range(n_inputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = X_ref[:, i]
            row[-1] = -X[k, i]
            A_ub_list.append(row)
            b_ub_list.append(0.0)

        for r in range(n_outputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = -Y_ref[:, r]
            A_ub_list.append(row)
            b_ub_list.append(float(-Y[k, r]))

    A_ub = np.vstack(A_ub_list) if A_ub_list else None
    b_ub = np.asarray(b_ub_list) if b_ub_list else None

    A_eq = None
    b_eq = None
    if returns == "vrs":
        A_eq = np.zeros((1, n_ref + 1))
        A_eq[0, :n_ref] = 1.0
        b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n_ref + [(0.0, None)]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.success:
        lambdas_ref = result.x[:n_ref]
        for pos, j in enumerate(ref_indices):
            full_lambdas[j] = lambdas_ref[pos]
        return {
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
            "factor": float(result.x[-1]),
            "lambdas": full_lambdas,
        }

    return {
        "success": False,
        "status": int(result.status),
        "message": str(result.message),
        "factor": np.nan,
        "lambdas": full_lambdas,
    }


def peer_string(lambdas: np.ndarray, dmu_ids: Sequence[str], max_peers: int = 6) -> str:
    peer_items = [
        (str(dmu_ids[j]), float(weight))
        for j, weight in enumerate(lambdas)
        if weight > PEER_TOL
    ]
    peer_items.sort(key=lambda item: item[1], reverse=True)
    if not peer_items:
        return "No peer identified"
    shown = peer_items[:max_peers]
    suffix = "" if len(peer_items) <= max_peers else f"; +{len(peer_items) - max_peers} more"
    return "; ".join([f"{name} ({weight:.3f})" for name, weight in shown]) + suffix


@st.cache_data(show_spinner=False)
def calculate_dea_with_slacks_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    return calculate_dea_with_slacks(data, list(input_cols), list(output_cols), orientation, returns)


def calculate_dea_with_slacks(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    """DEA with radial factor, 0-1 efficiency score, slacks, targets, peers, and diagnostics."""
    X = data[list(input_cols)].to_numpy(dtype=float)
    Y = data[list(output_cols)].to_numpy(dtype=float)
    n_units, n_inputs = X.shape
    n_outputs = Y.shape[1]

    efficiency = np.full(n_units, np.nan)
    radial_factor = np.full(n_units, np.nan)
    input_slacks = np.full((n_units, n_inputs), np.nan)
    output_slacks = np.full((n_units, n_outputs), np.nan)
    radial_input_reduction = np.zeros((n_units, n_inputs))
    radial_output_increase = np.zeros((n_units, n_outputs))
    target_inputs = np.full((n_units, n_inputs), np.nan)
    target_outputs = np.full((n_units, n_outputs), np.nan)
    lambdas = np.zeros((n_units, n_units))
    statuses: List[str] = []
    messages: List[str] = []

    for k in range(n_units):
        result = solve_dea_envelopment(X, Y, k, orientation=orientation, returns=returns, exclude_k=False)
        statuses.append("optimal" if result["success"] else f"status_{result['status']}")
        messages.append(str(result["message"]))

        if not result["success"]:
            continue

        factor = max(float(result["factor"]), TOL)
        lam = np.asarray(result["lambdas"], dtype=float)
        lambdas[k, :] = lam
        radial_factor[k] = factor

        X_proj = X.T @ lam
        Y_proj = Y.T @ lam

        if orientation == "output":
            # phi is an expansion factor. Conventional output-oriented efficiency is 1 / phi.
            score = 1.0 / factor
            efficiency[k] = float(np.clip(score, 0.0, 1.0))
            input_slacks[k, :] = np.maximum(X[k, :] - X_proj, 0.0)
            output_slacks[k, :] = np.maximum(Y_proj - factor * Y[k, :], 0.0)
            radial_output_increase[k, :] = np.maximum((factor - 1.0) * Y[k, :], 0.0)
        else:
            # theta is already a conventional 0-1 input-oriented efficiency score.
            efficiency[k] = float(np.clip(factor, 0.0, 1.0))
            input_slacks[k, :] = np.maximum(factor * X[k, :] - X_proj, 0.0)
            output_slacks[k, :] = np.maximum(Y_proj - Y[k, :], 0.0)
            radial_input_reduction[k, :] = np.maximum((1.0 - factor) * X[k, :], 0.0)

        target_inputs[k, :] = X_proj
        target_outputs[k, :] = Y_proj

    return {
        "efficiency": efficiency,
        "radial_factor": radial_factor,
        "input_slacks": input_slacks,
        "output_slacks": output_slacks,
        "radial_input_reduction": radial_input_reduction,
        "radial_output_increase": radial_output_increase,
        "target_inputs": target_inputs,
        "target_outputs": target_outputs,
        "lambdas": lambdas,
        "statuses": statuses,
        "messages": messages,
    }


@st.cache_data(show_spinner=False)
def calculate_super_efficiency_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    return calculate_super_efficiency(data, list(input_cols), list(output_cols), orientation, returns)


def calculate_super_efficiency(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    """Super-efficiency using leave-one-out DEA. Higher than 1 means beyond the peer frontier."""
    X = data[list(input_cols)].to_numpy(dtype=float)
    Y = data[list(output_cols)].to_numpy(dtype=float)
    n_units = len(data)

    scores = np.full(n_units, np.nan)
    statuses: List[str] = []
    messages: List[str] = []

    for k in range(n_units):
        result = solve_dea_envelopment(X, Y, k, orientation=orientation, returns=returns, exclude_k=True)
        statuses.append("optimal" if result["success"] else f"status_{result['status']}")
        messages.append(str(result["message"]))
        if not result["success"]:
            continue

        factor = max(float(result["factor"]), TOL)
        if orientation == "output":
            scores[k] = 1.0 / factor
        else:
            scores[k] = factor

    return {"super_efficiency": scores, "statuses": statuses, "messages": messages}


@st.cache_data(show_spinner=False)
def calculate_scale_diagnostics_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str,
) -> Dict[str, object]:
    return calculate_scale_diagnostics(data, list(input_cols), list(output_cols), orientation)


def calculate_scale_diagnostics(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str,
) -> Dict[str, object]:
    """Compute CRS, VRS, scale efficiency, and a lambda-sum RTS diagnostic."""
    crs = calculate_dea_with_slacks(data, input_cols, output_cols, orientation, "crs")
    vrs = calculate_dea_with_slacks(data, input_cols, output_cols, orientation, "vrs")

    crs_eff = np.asarray(crs["efficiency"], dtype=float)
    vrs_eff = np.asarray(vrs["efficiency"], dtype=float)
    scale_eff = np.full_like(crs_eff, np.nan)
    valid = np.isfinite(crs_eff) & np.isfinite(vrs_eff) & (vrs_eff > TOL)
    scale_eff[valid] = np.clip(crs_eff[valid] / vrs_eff[valid], 0.0, 1.0)

    lambda_sums = np.nansum(np.asarray(crs["lambdas"], dtype=float), axis=1)
    rts = []
    for se, lam_sum in zip(scale_eff, lambda_sums):
        if not np.isfinite(lam_sum) or not np.isfinite(se):
            rts.append("Unknown")
        elif abs(se - 1.0) <= 1e-4:
            rts.append("CRS")
        elif lam_sum < 1.0 - 1e-4:
            rts.append("IRS")
        elif lam_sum > 1.0 + 1e-4:
            rts.append("DRS")
        else:
            rts.append("Ambiguous")

    return {
        "crs_efficiency": crs_eff,
        "vrs_efficiency": vrs_eff,
        "scale_efficiency": scale_eff,
        "lambda_sum_crs": lambda_sums,
        "returns_to_scale": rts,
    }


def calculate_reference_support(
    lambdas: np.ndarray,
    efficiency: np.ndarray,
    dmu_ids: Sequence[str],
) -> pd.DataFrame:
    """Benchmark/reference-set support from envelopment lambdas."""
    n_units = lambdas.shape[0]
    rows = []
    for j in range(n_units):
        mask = lambdas[:, j] > PEER_TOL
        count = int(mask.sum())
        weight_sum = float(lambdas[:, j].sum())
        mean_eff_of_users = float(np.nanmean(efficiency[mask])) if count else np.nan
        rows.append(
            {
                "DMU_ID": str(dmu_ids[j]),
                "Reference_Count": count,
                "Reference_Weight_Sum": weight_sum,
                "Mean_Efficiency_of_DMUs_Using_Benchmark": mean_eff_of_users,
            }
        )
    support = pd.DataFrame(rows)
    max_count = support["Reference_Count"].max()
    support["Benchmark_Support_Score"] = np.where(
        max_count > 0, support["Reference_Count"] / max_count, 0.0
    )
    return support.sort_values(["Reference_Count", "Reference_Weight_Sum"], ascending=False)


@st.cache_data(show_spinner=False)
def calculate_cross_efficiency_crs_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    weight_floor: float = 0.0,
) -> Dict[str, object]:
    return calculate_cross_efficiency_crs(data, list(input_cols), list(output_cols), weight_floor)


def calculate_cross_efficiency_crs(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    weight_floor: float = 0.0,
) -> Dict[str, object]:
    """
    CCR multiplier cross-efficiency.

    Each DMU chooses output weights u and input weights v maximizing its own weighted
    output with v*x_k = 1 and u*y_j - v*x_j <= 0 for all DMUs. The resulting weights
    are then applied to every DMU as (u*y_j)/(v*x_j).
    """
    X = data[list(input_cols)].to_numpy(dtype=float)
    Y = data[list(output_cols)].to_numpy(dtype=float)
    n_units, n_inputs = X.shape
    n_outputs = Y.shape[1]

    matrix = np.full((n_units, n_units), np.nan)
    output_weights = np.full((n_units, n_outputs), np.nan)
    input_weights = np.full((n_units, n_inputs), np.nan)
    statuses: List[str] = []
    messages: List[str] = []

    for k in range(n_units):
        # Variables are [u_1...u_s, v_1...v_m].
        c = np.zeros(n_outputs + n_inputs)
        c[:n_outputs] = -Y[k, :]

        A_ub = []
        b_ub = []
        for j in range(n_units):
            row = np.zeros(n_outputs + n_inputs)
            row[:n_outputs] = Y[j, :]
            row[n_outputs:] = -X[j, :]
            A_ub.append(row)
            b_ub.append(0.0)

        A_eq = np.zeros((1, n_outputs + n_inputs))
        A_eq[0, n_outputs:] = X[k, :]
        b_eq = np.array([1.0])

        bounds = [(weight_floor, None)] * (n_outputs + n_inputs)
        result = linprog(
            c,
            A_ub=np.asarray(A_ub),
            b_ub=np.asarray(b_ub),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        statuses.append("optimal" if result.success else f"status_{result.status}")
        messages.append(str(result.message))
        if not result.success:
            continue

        u = result.x[:n_outputs]
        v = result.x[n_outputs:]
        output_weights[k, :] = u
        input_weights[k, :] = v
        numer = Y @ u
        denom = X @ v
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = numer / denom
        matrix[k, :] = np.where(np.isfinite(scores), np.clip(scores, 0.0, 1.0), np.nan)

    with np.errstate(invalid="ignore"):
        mean_cross = np.nanmean(matrix, axis=0)
        min_cross = np.nanmin(matrix, axis=0)
        max_cross = np.nanmax(matrix, axis=0)
        self_cross = np.diag(matrix)

    return {
        "matrix": matrix,
        "mean_cross": mean_cross,
        "min_cross": min_cross,
        "max_cross": max_cross,
        "self_cross": self_cross,
        "output_weights": output_weights,
        "input_weights": input_weights,
        "statuses": statuses,
        "messages": messages,
    }


@st.cache_data(show_spinner=False)
def calculate_jackknife_influence_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str,
    returns: str,
) -> pd.DataFrame:
    return calculate_jackknife_influence(data, list(input_cols), list(output_cols), orientation, returns)


def calculate_jackknife_influence(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str,
    returns: str,
) -> pd.DataFrame:
    """Leave-one-out influence: average absolute score change among remaining DMUs."""
    baseline = calculate_dea_with_slacks(data, input_cols, output_cols, orientation, returns)["efficiency"]
    dmu_ids = data["DMU_ID"].astype(str).tolist() if "DMU_ID" in data else [str(i) for i in range(len(data))]
    rows = []
    n = len(data)
    for omitted in range(n):
        reduced = data.drop(index=omitted).reset_index(drop=True)
        reduced_scores = calculate_dea_with_slacks(reduced, input_cols, output_cols, orientation, returns)["efficiency"]
        baseline_remaining = np.delete(baseline, omitted)
        mask = np.isfinite(reduced_scores) & np.isfinite(baseline_remaining)
        influence = float(np.mean(np.abs(reduced_scores[mask] - baseline_remaining[mask]))) if mask.any() else np.nan
        rows.append(
            {
                "Omitted_DMU_ID": dmu_ids[omitted],
                "Mean_Absolute_Change_in_Remaining_DEA_Scores": influence,
                "Baseline_DEA_Efficiency": baseline[omitted],
            }
        )
    return pd.DataFrame(rows).sort_values("Mean_Absolute_Change_in_Remaining_DEA_Scores", ascending=False)



# -----------------------------------------------------------------------------
# DEA bootstrap uncertainty
# -----------------------------------------------------------------------------


def solve_dea_against_reference(
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    """Evaluate one observed DMU against an externally supplied DEA reference set."""
    X_ref = np.asarray(X_ref, dtype=float)
    Y_ref = np.asarray(Y_ref, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float)

    n_ref, n_inputs = X_ref.shape
    n_outputs = Y_ref.shape[1]
    if n_ref == 0:
        return {
            "success": False,
            "status": -1,
            "message": "No bootstrap reference DMUs available.",
            "factor": np.nan,
            "score": np.nan,
            "raw_score": np.nan,
        }

    c = np.zeros(n_ref + 1)
    A_ub_list: List[np.ndarray] = []
    b_ub_list: List[float] = []

    if orientation == "output":
        # Maximize phi subject to X_ref lambda <= x_eval and Y_ref lambda >= phi*y_eval.
        # linprog minimizes, so minimize -phi. Report 1/phi as conventional efficiency.
        c[-1] = -1.0
        for i in range(n_inputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = X_ref[:, i]
            A_ub_list.append(row)
            b_ub_list.append(float(x_eval[i]))
        for r in range(n_outputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = -Y_ref[:, r]
            row[-1] = y_eval[r]
            A_ub_list.append(row)
            b_ub_list.append(0.0)
    else:
        # Minimize theta subject to X_ref lambda <= theta*x_eval and Y_ref lambda >= y_eval.
        c[-1] = 1.0
        for i in range(n_inputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = X_ref[:, i]
            row[-1] = -x_eval[i]
            A_ub_list.append(row)
            b_ub_list.append(0.0)
        for r in range(n_outputs):
            row = np.zeros(n_ref + 1)
            row[:n_ref] = -Y_ref[:, r]
            A_ub_list.append(row)
            b_ub_list.append(float(-y_eval[r]))

    A_eq = None
    b_eq = None
    if returns == "vrs":
        A_eq = np.zeros((1, n_ref + 1))
        A_eq[0, :n_ref] = 1.0
        b_eq = np.array([1.0])

    result = linprog(
        c,
        A_ub=np.vstack(A_ub_list),
        b_ub=np.asarray(b_ub_list),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n_ref + [(0.0, None)],
        method="highs",
    )

    if not result.success:
        return {
            "success": False,
            "status": int(result.status),
            "message": str(result.message),
            "factor": np.nan,
            "score": np.nan,
            "raw_score": np.nan,
        }

    factor = max(float(result.x[-1]), TOL)
    raw_score = 1.0 / factor if orientation == "output" else factor
    return {
        "success": True,
        "status": int(result.status),
        "message": str(result.message),
        "factor": factor,
        "score": float(np.clip(raw_score, 0.0, 1.0)),
        "raw_score": float(raw_score),
    }


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


def bootstrap_rank_matrix(score_matrix: np.ndarray) -> np.ndarray:
    """Rank each bootstrap draw, with higher score ranked better."""
    score_matrix = np.asarray(score_matrix, dtype=float)
    ranks = np.full_like(score_matrix, np.nan, dtype=float)
    for b in range(score_matrix.shape[0]):
        row = score_matrix[b, :]
        finite = np.isfinite(row)
        if finite.any():
            ranked = pd.Series(row).rank(ascending=False, na_option="bottom", method="min").to_numpy(dtype=float)
            ranked[~finite] = np.nan
            ranks[b, :] = ranked
    return ranks


@st.cache_data(show_spinner=False)
def calculate_dea_bootstrap_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str,
    returns: str,
    n_boot: int,
    ci_level: float,
    seed: int,
    include_self: bool,
) -> Dict[str, object]:
    return calculate_dea_bootstrap(
        data,
        list(input_cols),
        list(output_cols),
        orientation,
        returns,
        n_boot=int(n_boot),
        ci_level=float(ci_level),
        seed=int(seed),
        include_self=bool(include_self),
    )


def calculate_dea_bootstrap(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str,
    returns: str,
    n_boot: int = 200,
    ci_level: float = 95.0,
    seed: int = 12345,
    include_self: bool = True,
) -> Dict[str, object]:
    """
    Practical DEA frontier-resampling bootstrap.

    Each replication resamples DMUs with replacement, builds a bootstrap reference
    frontier, and evaluates every original DMU against that reference frontier.
    By default, the evaluated DMU is appended to its bootstrap reference set. That
    improves VRS feasibility and keeps conventional scores on a 0-1 scale, but the
    result should be treated as a robustness diagnostic rather than a full
    smoothed Simar-Wilson bootstrap.
    """
    n_boot = max(int(n_boot), 1)
    ci_level = float(np.clip(ci_level, 50.0, 99.9))
    alpha = (100.0 - ci_level) / 2.0

    X = data[list(input_cols)].to_numpy(dtype=float)
    Y = data[list(output_cols)].to_numpy(dtype=float)
    n_units = len(data)
    dmu_ids = data["DMU_ID"].astype(str).tolist() if "DMU_ID" in data else [f"DMU_{i + 1}" for i in range(n_units)]
    rng = np.random.default_rng(int(seed))

    baseline = np.asarray(
        calculate_dea_with_slacks(data, input_cols, output_cols, orientation, returns)["efficiency"],
        dtype=float,
    )
    scores = np.full((n_boot, n_units), np.nan)
    raw_scores = np.full((n_boot, n_units), np.nan)
    failures = np.zeros(n_units, dtype=int)

    for b in range(n_boot):
        sample_idx = rng.integers(0, n_units, size=n_units)
        X_ref_base = X[sample_idx, :]
        Y_ref_base = Y[sample_idx, :]

        for k in range(n_units):
            if include_self:
                X_ref = np.vstack([X_ref_base, X[k : k + 1, :]])
                Y_ref = np.vstack([Y_ref_base, Y[k : k + 1, :]])
            else:
                X_ref = X_ref_base
                Y_ref = Y_ref_base

            result = solve_dea_against_reference(
                X_ref,
                Y_ref,
                X[k, :],
                Y[k, :],
                orientation=orientation,
                returns=returns,
            )
            if result["success"]:
                scores[b, k] = float(result["score"])
                raw_scores[b, k] = float(result["raw_score"])
            else:
                failures[k] += 1

    ranks = bootstrap_rank_matrix(scores)
    boot_mean = column_mean(scores)
    boot_std = column_std(scores, ddof=1)
    rank_mean = column_mean(ranks)
    bias = boot_mean - baseline
    bias_corrected = np.clip(2.0 * baseline - boot_mean, 0.0, 1.0)
    valid_draws = np.isfinite(scores).sum(axis=0)

    table = pd.DataFrame(
        {
            "DMU_ID": dmu_ids,
            "DEA_Efficiency": baseline,
            "DEA_Bootstrap_Mean": boot_mean,
            "DEA_Bootstrap_Std_Error": boot_std,
            "DEA_Bootstrap_Bias": bias,
            "DEA_Bootstrap_Bias_Corrected": bias_corrected,
            "DEA_Bootstrap_CI_Lower": column_percentile(scores, alpha),
            "DEA_Bootstrap_CI_Median": column_percentile(scores, 50.0),
            "DEA_Bootstrap_CI_Upper": column_percentile(scores, 100.0 - alpha),
            "DEA_Bootstrap_Frontier_Rate": np.nanmean(scores >= 0.999, axis=0),
            "DEA_Bootstrap_RawScore_Above_1_Rate": np.nanmean(raw_scores > 1.0 + 1e-6, axis=0),
            "DEA_Bootstrap_Rank_Mean": rank_mean,
            "DEA_Bootstrap_Rank_Lower": column_percentile(ranks, alpha),
            "DEA_Bootstrap_Rank_Upper": column_percentile(ranks, 100.0 - alpha),
            "DEA_Bootstrap_Valid_Draws": valid_draws,
            "DEA_Bootstrap_LP_Success_Rate": 1.0 - failures / max(n_boot, 1),
        }
    )
    table["DEA_Bootstrap_CI_Width"] = table["DEA_Bootstrap_CI_Upper"] - table["DEA_Bootstrap_CI_Lower"]

    summary = {
        "method": "DEA frontier resampling with replacement; original DMUs evaluated against each bootstrap reference frontier",
        "replications_requested": int(n_boot),
        "replications_completed": int(n_boot),
        "confidence_level": ci_level,
        "orientation": orientation,
        "returns": returns,
        "seed": int(seed),
        "include_evaluated_dmu_in_reference": bool(include_self),
        "total_lp_failures": int(failures.sum()),
        "mean_lp_success_rate": float(np.nanmean(table["DEA_Bootstrap_LP_Success_Rate"])),
        "mean_ci_width": float(np.nanmean(table["DEA_Bootstrap_CI_Width"])),
        "mean_absolute_bias": float(np.nanmean(np.abs(bias))),
        "note": "Practical sensitivity interval, not a full smoothed Simar-Wilson or double-bootstrap estimator.",
    }
    return {"summary": summary, "table": table, "score_matrix": scores, "raw_score_matrix": raw_scores, "rank_matrix": ranks}

# -----------------------------------------------------------------------------
# SFA: stable half-normal production frontier
# -----------------------------------------------------------------------------


def build_sfa_design(log_X: np.ndarray, input_cols: Sequence[str], model_type: str) -> Tuple[np.ndarray, List[str]]:
    """Build Cobb-Douglas or Translog design matrix using logged inputs."""
    n_units, n_inputs = log_X.shape
    cols = [np.ones(n_units)]
    names = ["Intercept"]

    for i, col in enumerate(input_cols):
        cols.append(log_X[:, i])
        names.append(f"ln({col})")

    if model_type == "Translog":
        for i, col in enumerate(input_cols):
            cols.append(0.5 * log_X[:, i] ** 2)
            names.append(f"0.5*ln({col})^2")
        for i in range(n_inputs):
            for j in range(i + 1, n_inputs):
                cols.append(log_X[:, i] * log_X[:, j])
                names.append(f"ln({input_cols[i]})*ln({input_cols[j]})")

    return np.column_stack(cols), names


@st.cache_data(show_spinner=False)
def calculate_sfa_production_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_col: str,
    model_type: str = "Cobb-Douglas",
    frontier_type: str = "Production",
) -> Dict[str, object]:
    return calculate_sfa_production(data, list(input_cols), output_col, model_type, frontier_type)


def calculate_sfa_production(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: str,
    model_type: str = "Cobb-Douglas",
    frontier_type: str = "Production",
) -> Dict[str, object]:
    """Half-normal SFA estimated by maximum likelihood.

    Production frontier: log(y) = f(x) + v - u.
    Cost frontier: log(c) = f(x) + v + u.
    Efficiency is reported as E[exp(-u) | epsilon] for both.
    """
    n_units = len(data)
    X_raw = data[list(input_cols)].to_numpy(dtype=float)
    y_raw = data[output_col].to_numpy(dtype=float)

    empty_result: Dict[str, object] = {
        "efficiency": np.full(n_units, np.nan),
        "beta": np.array([]),
        "param_names": [],
        "std_errors": np.array([]),
        "sigma_u": np.nan,
        "sigma_v": np.nan,
        "lambda": np.nan,
        "gamma": np.nan,
        "log_likelihood": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "converged": False,
        "message": "SFA was not estimated.",
        "model_type": model_type,
        "frontier_type": frontier_type,
        "output_col": output_col,
    }

    if n_units < 5:
        empty_result["message"] = "SFA needs more observations for reliable estimation."
        return empty_result
    if np.any(X_raw <= 0) or np.any(y_raw <= 0):
        empty_result["message"] = "SFA requires strictly positive selected inputs and output."
        return empty_result

    log_X = np.log(X_raw)
    log_y = np.log(y_raw)
    X_reg, param_names = build_sfa_design(log_X, input_cols, model_type)
    n_params_beta = X_reg.shape[1]

    if n_units <= n_params_beta + 3:
        empty_result["message"] = (
            f"Not enough observations for {model_type} SFA: {n_units} rows and "
            f"{n_params_beta} frontier coefficients."
        )
        empty_result["param_names"] = param_names
        return empty_result

    try:
        beta_ols = np.linalg.lstsq(X_reg, log_y, rcond=None)[0]
        residuals_ols = log_y - X_reg @ beta_ols
        sigma0 = max(float(np.std(residuals_ols, ddof=min(n_params_beta, n_units - 1))), 1e-4)
    except np.linalg.LinAlgError as exc:
        empty_result["message"] = f"OLS initialization failed: {exc}"
        empty_result["param_names"] = param_names
        return empty_result

    frontier_type_norm = str(frontier_type).strip().lower()
    q = 1.0 if frontier_type_norm == "production" else -1.0

    def neg_loglik(params: np.ndarray) -> float:
        beta = params[:n_params_beta]
        sigma_u = math.exp(float(params[n_params_beta]))
        sigma_v = math.exp(float(params[n_params_beta + 1]))
        sigma = math.sqrt(sigma_u**2 + sigma_v**2)
        lam = sigma_u / sigma_v
        eps = log_y - X_reg @ beta
        z = eps / sigma
        loglik_i = np.log(2.0) - np.log(sigma) + norm.logpdf(z) + norm.logcdf(-q * lam * z)
        if not np.all(np.isfinite(loglik_i)):
            return 1e100
        return float(-np.sum(loglik_i))

    params0 = np.concatenate([beta_ols, [np.log(sigma0 / np.sqrt(2.0)), np.log(sigma0 / np.sqrt(2.0))]])
    bounds = [(None, None)] * n_params_beta + [(-20.0, 5.0), (-20.0, 5.0)]

    result = minimize(
        neg_loglik,
        params0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-6},
    )

    params = result.x if np.all(np.isfinite(result.x)) else params0
    nll = neg_loglik(params)
    if not np.isfinite(nll):
        empty_result["message"] = "SFA optimization produced a non-finite likelihood."
        empty_result["param_names"] = param_names
        return empty_result

    beta = params[:n_params_beta]
    sigma_u = math.exp(float(params[n_params_beta]))
    sigma_v = math.exp(float(params[n_params_beta + 1]))
    sigma_sq = sigma_u**2 + sigma_v**2
    sigma = math.sqrt(sigma_sq)
    lam = sigma_u / sigma_v
    gamma = sigma_u**2 / sigma_sq
    eps = log_y - X_reg @ beta

    # Conditional distribution of u | epsilon for half-normal frontier.
    mu_star = -q * eps * sigma_u**2 / sigma_sq
    sigma_star_sq = (sigma_u**2 * sigma_v**2) / sigma_sq
    sigma_star = math.sqrt(max(sigma_star_sq, TOL))
    denom_arg = mu_star / sigma_star
    log_te = (
        -mu_star
        + 0.5 * sigma_star_sq
        + norm.logcdf(denom_arg - sigma_star)
        - norm.logcdf(denom_arg)
    )
    te = np.exp(np.clip(log_te, -745, 0))
    te = np.clip(te, 0.0, 1.0)

    std_errors = np.full(len(params), np.nan)
    try:
        hess_inv = result.hess_inv
        if hasattr(hess_inv, "todense"):
            cov = np.asarray(hess_inv.todense(), dtype=float)
        else:
            cov = np.asarray(hess_inv, dtype=float)
        diag = np.diag(cov)
        std_errors = np.where(diag >= 0, np.sqrt(diag), np.nan)
    except Exception:
        pass

    log_likelihood = -nll
    k_total = len(params)
    aic = 2 * k_total - 2 * log_likelihood
    bic = np.log(n_units) * k_total - 2 * log_likelihood

    return {
        "efficiency": te,
        "beta": beta,
        "param_names": param_names,
        "std_errors": std_errors[:n_params_beta] if len(std_errors) >= n_params_beta else np.full(n_params_beta, np.nan),
        "all_param_std_errors": std_errors,
        "sigma_u": sigma_u,
        "sigma_v": sigma_v,
        "lambda": lam,
        "gamma": gamma,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "converged": bool(result.success),
        "message": str(result.message),
        "model_type": model_type,
        "frontier_type": frontier_type,
        "output_col": output_col,
    }



def calculate_sfa_efficiency_from_fit(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: str,
    model_type: str,
    frontier_type: str,
    beta: Sequence[float],
    sigma_u: float,
    sigma_v: float,
) -> np.ndarray:
    """Evaluate original observations using fitted half-normal SFA parameters."""
    n_units = len(data)
    beta_arr = np.asarray(beta, dtype=float)
    if len(beta_arr) == 0 or not np.isfinite(sigma_u) or not np.isfinite(sigma_v) or sigma_u <= 0 or sigma_v <= 0:
        return np.full(n_units, np.nan)

    X_raw = data[list(input_cols)].to_numpy(dtype=float)
    y_raw = data[output_col].to_numpy(dtype=float)
    if np.any(X_raw <= 0) or np.any(y_raw <= 0):
        return np.full(n_units, np.nan)

    X_reg, _ = build_sfa_design(np.log(X_raw), input_cols, model_type)
    if X_reg.shape[1] != len(beta_arr):
        return np.full(n_units, np.nan)

    log_y = np.log(y_raw)
    sigma_sq = float(sigma_u) ** 2 + float(sigma_v) ** 2
    if sigma_sq <= 0:
        return np.full(n_units, np.nan)

    frontier_type_norm = str(frontier_type).strip().lower()
    q = 1.0 if frontier_type_norm == "production" else -1.0
    eps = log_y - X_reg @ beta_arr
    mu_star = -q * eps * float(sigma_u) ** 2 / sigma_sq
    sigma_star_sq = (float(sigma_u) ** 2 * float(sigma_v) ** 2) / sigma_sq
    sigma_star = math.sqrt(max(sigma_star_sq, TOL))
    denom_arg = mu_star / sigma_star
    log_te = (
        -mu_star
        + 0.5 * sigma_star_sq
        + norm.logcdf(denom_arg - sigma_star)
        - norm.logcdf(denom_arg)
    )
    te = np.exp(np.clip(log_te, -745, 0))
    return np.clip(te, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def calculate_sfa_bootstrap_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_col: str,
    model_type: str,
    frontier_type: str,
    n_boot: int,
    ci_level: float,
    seed: int,
) -> Dict[str, object]:
    return calculate_sfa_bootstrap(
        data,
        list(input_cols),
        output_col,
        model_type,
        frontier_type,
        int(n_boot),
        float(ci_level),
        int(seed),
    )


def bootstrap_summary_table(
    names: Sequence[str],
    baseline: Sequence[float],
    samples: np.ndarray,
    prefix: str,
    ci_level: float = 95.0,
) -> pd.DataFrame:
    """Generic percentile interval table for bootstrap parameter or metric draws."""
    ci_level = float(np.clip(ci_level, 50.0, 99.9))
    alpha = (100.0 - ci_level) / 2.0
    baseline_arr = np.asarray(baseline, dtype=float)
    sample_arr = np.asarray(samples, dtype=float)
    if sample_arr.ndim != 2:
        sample_arr = np.empty((0, len(names)))

    rows = []
    for j, name in enumerate(names):
        vals = sample_arr[:, j] if sample_arr.shape[1] > j else np.array([], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            lower = float(np.percentile(vals, alpha))
            median = float(np.percentile(vals, 50.0))
            upper = float(np.percentile(vals, 100.0 - alpha))
        else:
            mean_val = std_val = lower = median = upper = np.nan
        rows.append(
            {
                f"{prefix}": str(name),
                "Baseline": baseline_arr[j] if len(baseline_arr) > j else np.nan,
                "Boot_Mean": mean_val,
                "Boot_Std": std_val,
                "Boot_CI_Lower": lower,
                "Boot_Median": median,
                "Boot_CI_Upper": upper,
                "Valid_Replications": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def calculate_sfa_bootstrap(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: str,
    model_type: str,
    frontier_type: str,
    n_boot: int,
    ci_level: float,
    seed: int,
) -> Dict[str, object]:
    """Pairs bootstrap for SFA coefficients, fit metrics, and fitted efficiencies.

    Each replication resamples rows with replacement, re-estimates the SFA model,
    and applies the fitted frontier parameters to the original rows so intervals are
    attached to the original DMUs.
    """
    n_boot = max(int(n_boot), 1)
    ci_level = float(np.clip(ci_level, 50.0, 99.9))
    alpha = (100.0 - ci_level) / 2.0
    n_units = len(data)
    rng = np.random.default_rng(int(seed))
    dmu_ids = data["DMU_ID"].astype(str).tolist() if "DMU_ID" in data else [f"DMU_{i + 1}" for i in range(n_units)]

    baseline_fit = calculate_sfa_production(data, input_cols, output_col, model_type, frontier_type)
    baseline_eff = np.asarray(baseline_fit.get("efficiency", np.full(n_units, np.nan)), dtype=float)
    param_names = list(baseline_fit.get("param_names", []))
    baseline_beta = np.asarray(baseline_fit.get("beta", np.full(len(param_names), np.nan)), dtype=float)
    metric_names = ["sigma_u", "sigma_v", "lambda", "gamma", "log_likelihood", "aic", "bic"]
    baseline_metrics = np.asarray([baseline_fit.get(name, np.nan) for name in metric_names], dtype=float)

    eff_scores = np.full((n_boot, n_units), np.nan)
    param_samples = np.full((n_boot, len(param_names)), np.nan)
    metric_samples = np.full((n_boot, len(metric_names)), np.nan)
    usable = 0
    converged = 0

    for b in range(n_boot):
        sample_idx = rng.integers(0, n_units, size=n_units)
        boot_data = data.iloc[sample_idx].reset_index(drop=True)
        fit = calculate_sfa_production(boot_data, input_cols, output_col, model_type, frontier_type)
        beta = np.asarray(fit.get("beta", []), dtype=float)
        sigma_u = float(fit.get("sigma_u", np.nan))
        sigma_v = float(fit.get("sigma_v", np.nan))
        if len(beta) != len(param_names) or not np.isfinite(sigma_u) or not np.isfinite(sigma_v) or sigma_u <= 0 or sigma_v <= 0:
            continue

        param_samples[b, :] = beta
        metric_samples[b, :] = np.asarray([fit.get(name, np.nan) for name in metric_names], dtype=float)
        eff_scores[b, :] = calculate_sfa_efficiency_from_fit(
            data,
            input_cols,
            output_col,
            model_type,
            frontier_type,
            beta,
            sigma_u,
            sigma_v,
        )
        usable += 1
        if bool(fit.get("converged", False)):
            converged += 1

    eff_mean = column_mean(eff_scores)
    eff_std = column_std(eff_scores, ddof=1)
    eff_bias = eff_mean - baseline_eff
    valid_eff = np.sum(np.isfinite(eff_scores), axis=0)
    efficiency_table = pd.DataFrame(
        {
            "DMU_ID": dmu_ids,
            "SFA_Efficiency": baseline_eff,
            "SFA_Boot_Mean": eff_mean,
            "SFA_Boot_Std": eff_std,
            "SFA_Boot_CI_Lower": column_percentile(eff_scores, alpha),
            "SFA_Boot_Median": column_percentile(eff_scores, 50.0),
            "SFA_Boot_CI_Upper": column_percentile(eff_scores, 100.0 - alpha),
            "SFA_Boot_Bias_MeanMinusBaseline": eff_bias,
            "SFA_Boot_Bias_Corrected": np.clip(baseline_eff - eff_bias, 0.0, 1.0),
            "SFA_Boot_Valid_Replications": valid_eff.astype(int),
        }
    )

    summary = {
        "method": "SFA pairs bootstrap: resample rows, re-estimate the frontier, evaluate original DMUs",
        "replications_requested": int(n_boot),
        "usable_replications": int(usable),
        "converged_replications": int(converged),
        "model_type": model_type,
        "frontier_type": frontier_type,
        "output_col": output_col,
        "confidence_level": ci_level,
        "seed": int(seed),
        "note": "Intervals reflect case-resampling variability and may be unstable if the SFA likelihood is weakly identified.",
    }

    return {
        "summary": summary,
        "efficiency_table": efficiency_table,
        "parameter_table": bootstrap_summary_table(param_names, baseline_beta, param_samples, "Parameter", ci_level),
        "metric_table": bootstrap_summary_table(metric_names, baseline_metrics, metric_samples, "Metric", ci_level),
        "efficiency_matrix": eff_scores,
        "parameter_matrix": param_samples,
        "metric_matrix": metric_samples,
    }


# -----------------------------------------------------------------------------
# Result tables and exports
# -----------------------------------------------------------------------------


def build_target_table(
    df_clean: pd.DataFrame,
    dea: Dict[str, object],
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    dmu_ids: Sequence[str],
) -> pd.DataFrame:
    """Create managerial DEA target table with current, target, absolute, and percent changes."""
    efficiency = np.asarray(dea["efficiency"], dtype=float)
    target_inputs = np.asarray(dea["target_inputs"], dtype=float)
    target_outputs = np.asarray(dea["target_outputs"], dtype=float)
    lambdas = np.asarray(dea["lambdas"], dtype=float)
    input_slacks = np.asarray(dea["input_slacks"], dtype=float)
    output_slacks = np.asarray(dea["output_slacks"], dtype=float)

    table = pd.DataFrame({"DMU_ID": list(dmu_ids), "DEA_Efficiency": efficiency})
    table["DEA_Peers"] = [peer_string(lambdas[k, :], dmu_ids) for k in range(len(dmu_ids))]

    for i, col in enumerate(input_cols):
        current = df_clean[col].to_numpy(dtype=float)
        target = target_inputs[:, i]
        reduction = current - target
        table[f"Current_Input_{col}"] = current
        table[f"Target_Input_{col}"] = target
        table[f"Total_Input_Reduction_{col}"] = reduction
        table[f"Total_Input_Reduction_%_{col}"] = np.where(current > 0, 100.0 * reduction / current, np.nan)
        table[f"NonRadial_Input_Slack_{col}"] = input_slacks[:, i]

    for r, col in enumerate(output_cols):
        current = df_clean[col].to_numpy(dtype=float)
        target = target_outputs[:, r]
        increase = target - current
        table[f"Current_Output_{col}"] = current
        table[f"Target_Output_{col}"] = target
        table[f"Total_Output_Increase_{col}"] = increase
        table[f"Total_Output_Increase_%_{col}"] = np.where(current > 0, 100.0 * increase / current, np.nan)
        table[f"NonRadial_Output_Slack_{col}"] = output_slacks[:, r]

    return table


def build_peer_weights_table(lambdas: np.ndarray, dmu_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for k, dmu in enumerate(dmu_ids):
        for j, peer in enumerate(dmu_ids):
            weight = float(lambdas[k, j])
            if weight > PEER_TOL:
                rows.append({"DMU_ID": str(dmu), "Peer_DMU_ID": str(peer), "Lambda_Weight": weight})
    return pd.DataFrame(rows)


def build_results_table(
    df_clean: pd.DataFrame,
    dea: Dict[str, object],
    scale: Dict[str, object],
    sfa: Dict[str, object],
    super_eff: Optional[Dict[str, object]],
    support: pd.DataFrame,
    cross_eff: Optional[Dict[str, object]],
    metadata_cols: Sequence[str],
    input_cols: Sequence[str],
    output_cols: Sequence[str],
) -> pd.DataFrame:
    dmu_ids = df_clean["DMU_ID"].astype(str).tolist()
    keep_cols = unique_keep_order(["DMU_ID", "Original_Row"] + list(metadata_cols) + list(input_cols) + list(output_cols))
    keep_cols = [col for col in keep_cols if col in df_clean.columns]
    results = df_clean.loc[:, keep_cols].copy()

    results["DEA_Efficiency"] = np.asarray(dea["efficiency"], dtype=float)
    results["DEA_Radial_Factor"] = np.asarray(dea["radial_factor"], dtype=float)
    results["DEA_Rank"] = safe_rank(results["DEA_Efficiency"], ascending=False)
    results["DEA_Status"] = dea["statuses"]
    results["DEA_Message"] = dea["messages"]
    results["DEA_Peers"] = [peer_string(np.asarray(dea["lambdas"])[k, :], dmu_ids) for k in range(len(dmu_ids))]

    results["DEA_CRS_Efficiency"] = np.asarray(scale["crs_efficiency"], dtype=float)
    results["DEA_VRS_Efficiency"] = np.asarray(scale["vrs_efficiency"], dtype=float)
    results["Scale_Efficiency"] = np.asarray(scale["scale_efficiency"], dtype=float)
    results["CRS_Lambda_Sum"] = np.asarray(scale["lambda_sum_crs"], dtype=float)
    results["Returns_to_Scale"] = list(scale["returns_to_scale"])

    if super_eff is not None:
        results["DEA_SuperEfficiency"] = np.asarray(super_eff["super_efficiency"], dtype=float)
        results["DEA_SuperEfficiency_Rank"] = safe_rank(results["DEA_SuperEfficiency"], ascending=False)
        results["DEA_SuperEfficiency_Status"] = super_eff["statuses"]

    results["SFA_Efficiency"] = np.asarray(sfa["efficiency"], dtype=float)
    results["SFA_Rank"] = safe_rank(results["SFA_Efficiency"], ascending=False)

    support_for_merge = support[["DMU_ID", "Reference_Count", "Reference_Weight_Sum", "Benchmark_Support_Score"]]
    results = results.merge(support_for_merge, on="DMU_ID", how="left")

    if cross_eff is not None:
        results["CCR_Cross_Efficiency_Mean"] = np.asarray(cross_eff["mean_cross"], dtype=float)
        results["CCR_Cross_Efficiency_Min"] = np.asarray(cross_eff["min_cross"], dtype=float)
        results["CCR_Cross_Efficiency_Max"] = np.asarray(cross_eff["max_cross"], dtype=float)
        results["CCR_Self_Efficiency_from_Multiplier"] = np.asarray(cross_eff["self_cross"], dtype=float)
        results["Self_vs_Cross_Gap"] = results["DEA_Efficiency"] - results["CCR_Cross_Efficiency_Mean"]

    return results


def make_excel_download(tables: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)
    return buffer.getvalue()


def sfa_parameters_table(sfa: Dict[str, object]) -> pd.DataFrame:
    names = list(sfa.get("param_names", []))
    beta = np.asarray(sfa.get("beta", []), dtype=float)
    se = np.asarray(sfa.get("std_errors", []), dtype=float)

    if not names and len(beta) == 0:
        return pd.DataFrame(columns=["Parameter", "Coefficient", "Approx_Std_Error", "z_statistic"])

    if len(beta) != len(names):
        beta = np.full(len(names), np.nan)
    if len(se) != len(names):
        se = np.full(len(names), np.nan)

    table = pd.DataFrame({"Parameter": names, "Coefficient": beta, "Approx_Std_Error": se})
    with np.errstate(divide="ignore", invalid="ignore"):
        table["z_statistic"] = table["Coefficient"] / table["Approx_Std_Error"]
    return table


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def render_intro() -> None:
    st.info("Upload a CSV or Excel file to get started.")
    st.markdown(
        """
        ### What this version adds

        **DEA:** corrected output-oriented 0-1 efficiency scores, slacks, target input/output levels,
        peer weights, super-efficiency, CRS/VRS scale efficiency, returns-to-scale diagnostics,
        solver messages, and benchmark support.

        **SFA:** explicit single-output production or cost frontier, Cobb-Douglas or Translog specification,
        half-normal ML estimation, convergence diagnostics, gamma, AIC/BIC, and coefficient output.

        **Workflow:** DMU identifiers are preserved, optional metadata columns can be exported,
        expensive computations are cached, optional DEA bootstrap robustness is available,
        and downloads include results, targets, peers, bootstrap summaries, and diagnostics.
        """
    )


def render_sidebar(df: pd.DataFrame) -> Dict[str, object]:
    st.sidebar.header("Configuration")
    all_cols = list(df.columns)
    numeric_candidates = likely_numeric_columns(df)

    id_col = add_sidebar_selectbox("DMU identifier column", all_cols, "Optional unit name or ID column.")
    time_col = add_sidebar_selectbox("Time column", all_cols, "Optional year, quarter, month, or period column.")
    group_col = add_sidebar_selectbox("Group / peer-set column", all_cols, "Optional group column for filtering.")

    selected_groups: Optional[List[object]] = None
    if group_col is not None:
        group_values = sorted(df[group_col].dropna().unique().tolist(), key=lambda x: str(x))
        selected_groups = st.sidebar.multiselect(
            "Groups to include",
            options=group_values,
            default=group_values,
            help="Restrict analysis to selected peer groups. DEA should compare reasonably similar DMUs.",
        )

    st.sidebar.markdown("---")
    input_cols = st.sidebar.multiselect(
        "Input columns",
        options=numeric_candidates,
        help="Resources or costs to minimize, such as labor, beds, expenditure, or capital.",
    )
    output_cols = st.sidebar.multiselect(
        "Output columns",
        options=numeric_candidates,
        help="Services or outcomes to maximize. DEA can use multiple outputs; SFA below uses one output.",
    )

    env_options = [col for col in numeric_candidates if col not in set(input_cols).union(output_cols)]
    env_cols = st.sidebar.multiselect(
        "Environmental / case-mix columns to preserve",
        options=env_options,
        help="Optional columns kept in exports for stratification or second-stage analysis. They are not treated as discretionary DEA inputs/outputs.",
    )

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        dea_orientation = st.selectbox("DEA orientation", ["output", "input"])
    with col2:
        dea_returns = st.selectbox("DEA returns", ["crs", "vrs"])

    sfa_model = st.sidebar.selectbox("SFA frontier form", ["Cobb-Douglas", "Translog"])
    sfa_frontier_type = st.sidebar.selectbox("SFA frontier type", ["Production", "Cost"])
    sfa_output_col = output_cols[0] if output_cols else None
    if len(output_cols) > 1:
        sfa_output_col = st.sidebar.selectbox(
            "SFA output column",
            options=output_cols,
            help="SFA production frontier is single-output in this app. DEA still uses all selected outputs.",
        )

    st.sidebar.markdown("---")
    run_super_eff = st.sidebar.checkbox("Run super-efficiency", value=True)
    run_cross_eff = st.sidebar.checkbox(
        "Run CCR multiplier cross-efficiency",
        value=False,
        help="Solves one multiplier DEA model per DMU. It is CRS/CCR weight-based and may be slower on large data.",
    )
    run_jackknife = st.sidebar.checkbox(
        "Run leave-one-out robustness",
        value=False,
        help="Runs many additional DEA models. Best for small or medium datasets.",
    )

    with st.sidebar.expander("Bootstrap options", expanded=False):
        run_dea_bootstrap = st.checkbox(
            "Run DEA bootstrap intervals",
            value=False,
            help=(
                "Resamples DMUs with replacement, rebuilds a DEA reference frontier, "
                "and evaluates the original DMUs against each resampled frontier."
            ),
        )
        dea_bootstrap_replications = st.slider(
            "DEA bootstrap replications",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="More replications are more stable but require many additional linear programs.",
        )
        run_sfa_bootstrap = st.checkbox(
            "Run SFA bootstrap intervals",
            value=False,
            help="Pairs bootstrap for SFA coefficients and fitted efficiency intervals. This can be slower than DEA scoring.",
        )
        sfa_bootstrap_replications = st.slider(
            "SFA bootstrap replications",
            min_value=25,
            max_value=300,
            value=100,
            step=25,
            help="Each replication re-estimates the SFA model on a resampled dataset.",
        )
        bootstrap_ci_level = st.selectbox("Bootstrap CI level", options=[90, 95, 99], index=1)
        dea_bootstrap_include_self = st.checkbox(
            "Append evaluated DMU to each DEA bootstrap frontier",
            value=True,
            help="Improves VRS feasibility and keeps conventional DEA scores on a 0-1 scale. Turn off for a stricter resampled-frontier stress test.",
        )
        bootstrap_seed = st.number_input(
            "Bootstrap random seed",
            min_value=0,
            max_value=2147483647,
            value=12345,
            step=1,
        )

    return {
        "id_col": id_col,
        "time_col": time_col,
        "group_col": group_col,
        "selected_groups": selected_groups,
        "input_cols": input_cols,
        "output_cols": output_cols,
        "env_cols": env_cols,
        "dea_orientation": dea_orientation,
        "dea_returns": dea_returns,
        "sfa_model": sfa_model,
        "sfa_frontier_type": sfa_frontier_type,
        "sfa_output_col": sfa_output_col,
        "run_super_eff": run_super_eff,
        "run_cross_eff": run_cross_eff,
        "run_jackknife": run_jackknife,
        "run_dea_bootstrap": run_dea_bootstrap,
        "dea_bootstrap_replications": int(dea_bootstrap_replications),
        "bootstrap_ci_level": float(bootstrap_ci_level),
        "dea_bootstrap_include_self": bool(dea_bootstrap_include_self),
        "run_sfa_bootstrap": run_sfa_bootstrap,
        "sfa_bootstrap_replications": int(sfa_bootstrap_replications),
        "bootstrap_seed": int(bootstrap_seed),
    }


def render_validation(summary: Dict[str, object], warnings_list: Sequence[str]) -> None:
    removed_missing = int(summary["rows_removed_missing"])
    removed_nonpositive = int(summary["rows_removed_nonpositive"])
    if removed_missing or removed_nonpositive:
        st.warning(
            f"Removed {removed_missing + removed_nonpositive} rows before modeling: "
            f"{removed_missing} with missing/non-numeric model values and "
            f"{removed_nonpositive} with non-positive model values."
        )
    if int(summary.get("duplicate_ids", 0)) > 0:
        st.warning(f"Found {summary['duplicate_ids']} duplicate DMU IDs after cleaning.")
    for warning in warnings_list:
        st.warning(warning)


def render_results_tab(results_df: pd.DataFrame, dea_scores: np.ndarray, sfa_scores: np.ndarray) -> None:
    st.subheader("Efficiency Scores and Rankings")
    with st.expander("How to read these results", expanded=False):
        st.markdown(
            """
            **DEA_Efficiency and SFA_Efficiency are reported on a 0-1 scale**, where 1 means the unit is on the estimated frontier.

            For output-oriented DEA, the LP estimates an output expansion factor phi. This app reports **1 / phi** so the score has the same interpretation as input-oriented DEA.

            **Super-efficiency**, when available, can exceed 1 because it removes the evaluated unit from the reference set to rank frontier units.
            """
        )

    display_cols = [
        col
        for col in [
            "DMU_ID",
            "DEA_Efficiency",
            "DEA_Rank",
            "DEA_SuperEfficiency",
            "DEA_SuperEfficiency_Rank",
            "SFA_Efficiency",
            "SFA_Rank",
            "DEA_Bootstrap_CI_Lower",
            "DEA_Bootstrap_CI_Upper",
            "SFA_Boot_CI_Lower",
            "SFA_Boot_CI_Upper",
            "Scale_Efficiency",
            "Returns_to_Scale",
            "Reference_Count",
            "Benchmark_Support_Score",
        ]
        if col in results_df.columns
    ]
    st.dataframe(results_df[display_cols].round(4), width="stretch")

    valid_dea = dea_scores[np.isfinite(dea_scores)]
    valid_sfa = sfa_scores[np.isfinite(sfa_scores)]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DMUs analyzed", f"{len(results_df):,}")
    col2.metric("Avg DEA", f"{np.nanmean(valid_dea):.4f}" if len(valid_dea) else "NA")
    col3.metric("Frontier DMUs", f"{int(np.nansum(dea_scores >= 0.999)):,}")
    col4.metric("Avg SFA", f"{np.nanmean(valid_sfa):.4f}" if len(valid_sfa) else "NA")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Top DEA performers")
        top_cols = [c for c in ["DMU_ID", "DEA_Efficiency", "DEA_SuperEfficiency", "Scale_Efficiency"] if c in results_df]
        st.dataframe(results_df.nlargest(min(10, len(results_df)), "DEA_Efficiency")[top_cols].round(4), width="stretch")
    with col_right:
        st.subheader("Largest DEA improvement opportunities")
        bottom_cols = [c for c in ["DMU_ID", "DEA_Efficiency", "SFA_Efficiency", "Returns_to_Scale"] if c in results_df]
        st.dataframe(results_df.nsmallest(min(10, len(results_df)), "DEA_Efficiency")[bottom_cols].round(4), width="stretch")


def render_targets_tab(target_df: pd.DataFrame, peer_weights: pd.DataFrame, dmu_ids: Sequence[str]) -> None:
    st.subheader("DEA Targets, Slacks, and Peers")
    with st.expander("What this table shows", expanded=False):
        st.markdown(
            """
            DEA targets translate efficiency scores into operational changes. Input-oriented models emphasize feasible input reductions; output-oriented models emphasize feasible output increases. Non-radial slack is the extra adjustment after the radial movement.
            """
        )
    st.dataframe(target_df.round(4), width="stretch")

    selected_dmu = st.selectbox("Inspect one DMU", options=list(dmu_ids))
    selected_row = target_df[target_df["DMU_ID"] == selected_dmu]
    if not selected_row.empty:
        st.write("**Peer set**")
        st.write(selected_row.iloc[0].get("DEA_Peers", "No peer identified"))
        dmu_peer_weights = peer_weights[peer_weights["DMU_ID"] == selected_dmu]
        if not dmu_peer_weights.empty:
            st.dataframe(dmu_peer_weights.round(6), width="stretch")
        else:
            st.info("No positive peer weights were returned for this unit.")


def render_comparison_tab(results_df: pd.DataFrame) -> None:
    st.subheader("DEA versus SFA")
    dea_scores = results_df["DEA_Efficiency"].to_numpy(dtype=float)
    sfa_scores = results_df["SFA_Efficiency"].to_numpy(dtype=float)
    corr = finite_corr(dea_scores, sfa_scores)
    rank_corr = finite_spearman(dea_scores, sfa_scores)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pearson score corr.", f"{corr:.3f}" if np.isfinite(corr) else "NA")
    col2.metric("Spearman rank corr.", f"{rank_corr:.3f}" if np.isfinite(rank_corr) else "NA")
    col3.metric("DEA std. dev.", f"{np.nanstd(dea_scores):.4f}")
    col4.metric("SFA std. dev.", f"{np.nanstd(sfa_scores):.4f}" if np.isfinite(sfa_scores).any() else "NA")

    mask = np.isfinite(dea_scores) & np.isfinite(sfa_scores)
    if mask.sum() >= 3:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(dea_scores[mask], sfa_scores[mask], alpha=0.70)
        ax.set_xlabel("DEA efficiency")
        ax.set_ylabel("SFA efficiency")
        ax.set_title("DEA vs SFA efficiency")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        agreement_value = rank_corr if np.isfinite(rank_corr) else corr
        if agreement_value > 0.80:
            st.success(f"Strong agreement ({agreement_value:.3f}). DEA and SFA tell a similar ranking story.")
        elif agreement_value > 0.60:
            st.info(f"Moderate agreement ({agreement_value:.3f}). Review assumptions and target differences.")
        elif np.isfinite(agreement_value):
            st.warning(f"Low agreement ({agreement_value:.3f}). Noise, outliers, case-mix, or model form may matter.")
    else:
        st.info("SFA scores are unavailable or insufficient for a DEA/SFA comparison.")


def render_distributions_tab(results_df: pd.DataFrame) -> None:
    st.subheader("Efficiency Distributions")
    for col in ["DEA_Efficiency", "SFA_Efficiency", "Scale_Efficiency"]:
        if col not in results_df:
            continue
        values = results_df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(values, bins=min(20, max(5, int(np.sqrt(len(values))))), edgecolor="black", alpha=0.75)
        ax.axvline(np.mean(values), linestyle="--", linewidth=2, label="Mean")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{col} distribution")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


def render_frontier_tab(df_clean: pd.DataFrame, input_cols: Sequence[str], output_cols: Sequence[str], results_df: pd.DataFrame) -> None:
    st.subheader("Frontier Visualization")
    if len(input_cols) == 1 and len(output_cols) == 1:
        x = df_clean[input_cols[0]].to_numpy(dtype=float)
        y = df_clean[output_cols[0]].to_numpy(dtype=float)
        eff = results_df["DEA_Efficiency"].to_numpy(dtype=float)
        frontier_mask = np.isfinite(eff) & (eff >= 0.999)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(x, y, c=eff, s=80, alpha=0.80, edgecolors="black")
        if frontier_mask.any():
            ax.scatter(x[frontier_mask], y[frontier_mask], marker="*", s=220, edgecolors="black", label="DEA frontier")
        ax.set_xlabel(input_cols[0])
        ax.set_ylabel(output_cols[0])
        ax.set_title("Observed production set")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, label="DEA efficiency")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Frontier visualization is shown when exactly one input and one output are selected.")


def render_scale_tab(results_df: pd.DataFrame) -> None:
    st.subheader("Scale Efficiency and Returns to Scale")
    scale_cols = ["DMU_ID", "DEA_CRS_Efficiency", "DEA_VRS_Efficiency", "Scale_Efficiency", "CRS_Lambda_Sum", "Returns_to_Scale"]
    st.dataframe(results_df[scale_cols].round(4), width="stretch")

    counts = results_df["Returns_to_Scale"].value_counts(dropna=False)
    if not counts.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Returns-to-scale diagnostic")
        ax.set_ylabel("Number of DMUs")
        ax.set_title("Returns-to-scale profile")
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "RTS is classified from the CRS reference-set lambda sum: below 1 = IRS, near 1 = CRS, above 1 = DRS. Treat this as a diagnostic, especially when multiple optima are possible."
    )


def render_benchmark_tab(
    support: pd.DataFrame,
    cross_eff: Optional[Dict[str, object]],
    dmu_ids: Sequence[str],
) -> None:
    st.subheader("Benchmark Support and Cross-Efficiency")
    st.markdown(
        "Benchmark support counts how often each DMU appears in other units' DEA reference sets. This is not the same as multiplier cross-efficiency."
    )
    st.dataframe(support.round(4), width="stretch")

    if cross_eff is None:
        st.info("CCR multiplier cross-efficiency was not run. Enable it in the sidebar to solve weight-based peer appraisal models.")
        return

    cross_summary = pd.DataFrame(
        {
            "DMU_ID": list(dmu_ids),
            "Self_Efficiency_from_Multiplier": np.asarray(cross_eff["self_cross"], dtype=float),
            "Mean_CCR_Cross_Efficiency": np.asarray(cross_eff["mean_cross"], dtype=float),
            "Min_CCR_Cross_Efficiency": np.asarray(cross_eff["min_cross"], dtype=float),
            "Max_CCR_Cross_Efficiency": np.asarray(cross_eff["max_cross"], dtype=float),
        }
    )
    st.write("**CCR multiplier cross-efficiency summary**")
    st.dataframe(cross_summary.round(4), width="stretch")

    matrix = pd.DataFrame(cross_eff["matrix"], index=dmu_ids, columns=dmu_ids)
    with st.expander("Cross-efficiency matrix: rows are evaluator weights, columns are evaluated DMUs", expanded=False):
        st.dataframe(matrix.round(4), width="stretch")


def render_super_and_robustness_tab(
    results_df: pd.DataFrame,
    jackknife_df: Optional[pd.DataFrame],
    dea_bootstrap: Optional[Dict[str, object]],
    sfa_bootstrap: Optional[Dict[str, object]],
    orientation: str,
    returns: str,
) -> None:
    st.subheader("Super-Efficiency, Bootstrapping, and Robustness")

    if "DEA_SuperEfficiency" in results_df:
        cols = ["DMU_ID", "DEA_Efficiency", "DEA_SuperEfficiency", "DEA_SuperEfficiency_Rank", "DEA_SuperEfficiency_Status"]
        st.write("**Super-efficiency ranking**")
        st.dataframe(results_df.sort_values("DEA_SuperEfficiency", ascending=False)[cols].round(4), width="stretch")
    else:
        st.info("Super-efficiency was not run. Enable it in the sidebar.")

    st.write("**Deterministic robustness checks**")
    corr_scale = finite_corr(results_df["DEA_CRS_Efficiency"], results_df["DEA_VRS_Efficiency"])
    corr_dea_scale = finite_corr(results_df["DEA_Efficiency"], results_df["Scale_Efficiency"])
    rank_corr_dea_sfa = finite_spearman(results_df["DEA_Efficiency"], results_df["SFA_Efficiency"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CRS/VRS score correlation", f"{corr_scale:.3f}" if np.isfinite(corr_scale) else "NA")
    col2.metric("DEA/scale correlation", f"{corr_dea_scale:.3f}" if np.isfinite(corr_dea_scale) else "NA")
    col3.metric("DEA/SFA rank correlation", f"{rank_corr_dea_sfa:.3f}" if np.isfinite(rank_corr_dea_sfa) else "NA")
    col4.metric("Selected model", f"{orientation.upper()} / {returns.upper()}")

    st.write("**Bootstrap robustness intervals**")
    if dea_bootstrap is not None:
        st.caption("DEA bootstrap uses frontier resampling: each draw resamples DMUs, rebuilds the DEA reference frontier, and evaluates original DMUs against that resampled frontier.")
        st.json(dea_bootstrap.get("summary", {}))
        dea_boot_df = dea_bootstrap.get("table", pd.DataFrame())
        if isinstance(dea_boot_df, pd.DataFrame) and not dea_boot_df.empty:
            st.write("DEA efficiency and rank intervals")
            st.dataframe(dea_boot_df.round(4), width="stretch")

            plot_df = dea_boot_df.sort_values("DEA_Efficiency", ascending=False).head(min(25, len(dea_boot_df))).copy()
            y = plot_df["DEA_Bootstrap_Mean"].to_numpy(dtype=float)
            lower = plot_df["DEA_Bootstrap_CI_Lower"].to_numpy(dtype=float)
            upper = plot_df["DEA_Bootstrap_CI_Upper"].to_numpy(dtype=float)
            x = np.arange(len(plot_df))
            yerr = np.vstack([np.maximum(y - lower, 0.0), np.maximum(upper - y, 0.0)])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["DMU_ID"].astype(str), rotation=90)
            ax.set_ylabel("DEA bootstrap efficiency")
            ax.set_title("DEA bootstrap intervals for top baseline DEA performers")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("DEA bootstrap intervals were not run. Enable them in the sidebar Bootstrap options.")

    if sfa_bootstrap is not None:
        st.caption("SFA bootstrap is a pairs bootstrap: each draw resamples rows, re-estimates the SFA model, and evaluates fitted efficiencies for the original DMUs.")
        st.json(sfa_bootstrap.get("summary", {}))
        param_table = sfa_bootstrap.get("parameter_table", pd.DataFrame())
        metric_table = sfa_bootstrap.get("metric_table", pd.DataFrame())
        eff_table = sfa_bootstrap.get("efficiency_table", pd.DataFrame())
        if isinstance(param_table, pd.DataFrame) and not param_table.empty:
            st.write("SFA coefficient bootstrap intervals")
            st.dataframe(param_table.round(6), width="stretch")
        if isinstance(metric_table, pd.DataFrame) and not metric_table.empty:
            st.write("SFA variance and fit metric bootstrap intervals")
            st.dataframe(metric_table.round(6), width="stretch")
        if isinstance(eff_table, pd.DataFrame) and not eff_table.empty:
            st.write("SFA efficiency bootstrap intervals")
            st.dataframe(eff_table.round(4), width="stretch")
    else:
        st.info("SFA bootstrap intervals were not run. Enable them in the sidebar Bootstrap options.")

    if jackknife_df is not None:
        st.write("**Leave-one-out influence**")
        st.dataframe(jackknife_df.round(6), width="stretch")
    else:
        st.info("Leave-one-out robustness was not run. Enable it in the sidebar for small and medium datasets.")

    st.caption(
        "Bootstrap intervals are intended for robustness screening. For publication-grade DEA inference, consider a dedicated Simar-Wilson smoothed bootstrap or double-bootstrap workflow."
    )

def render_sfa_tab(sfa: Dict[str, object], sfa_param_df: pd.DataFrame) -> None:
    st.subheader("SFA Details")
    if not sfa.get("converged", False):
        st.warning(f"SFA convergence warning: {sfa.get('message', 'No message')}")
    else:
        st.success("SFA optimizer converged.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Output", str(sfa.get("output_col", "NA")))
    st.caption(f"Frontier type: {sfa.get('frontier_type', 'Production')}")
    col2.metric("Gamma", f"{sfa.get('gamma', np.nan):.4f}" if np.isfinite(sfa.get("gamma", np.nan)) else "NA")
    col3.metric("Log likelihood", f"{sfa.get('log_likelihood', np.nan):.2f}" if np.isfinite(sfa.get("log_likelihood", np.nan)) else "NA")
    col4.metric("AIC", f"{sfa.get('aic', np.nan):.2f}" if np.isfinite(sfa.get("aic", np.nan)) else "NA")

    with st.expander("**Variance decomposition**"):
        variance_table = pd.DataFrame(
            {
                "Metric": ["sigma_u", "sigma_v", "lambda_sigma_u_over_sigma_v", "gamma", "BIC"],
                "Value": [
                    sfa.get("sigma_u", np.nan),
                    sfa.get("sigma_v", np.nan),
                    sfa.get("lambda", np.nan),
                    sfa.get("gamma", np.nan),
                    sfa.get("bic", np.nan),
                ],
            }
        )
        st.dataframe(variance_table.round(6), width="content")

    with st.expander("**Frontier coefficients**"):
        if not sfa_param_df.empty:
            st.dataframe(sfa_param_df.round(6), width="stretch")
        else:
            st.info("No SFA parameters are available.")

        st.caption(
            "The approximate standard errors come from the optimizer's inverse-Hessian approximation and should be treated as diagnostic rather than publication-ready."
        )


def render_diagnostics_tab(
    summary: Dict[str, object],
    errors: Sequence[str],
    warnings_list: Sequence[str],
    results_df: pd.DataFrame,
) -> None:
    st.subheader("Data and Solver Diagnostics")
    with st.expander("**Cleaning summary**"):
        st.dataframe(summary, width="content")

        if errors:
            st.error("Blocking setup errors: " + " | ".join(errors))
        if warnings_list:
            st.write("**Warnings**")
            for warning in warnings_list:
                st.warning(warning)
        else:
            st.success("No validation warnings were generated.")

    solver_cols = [c for c in ["DMU_ID", "DEA_Status", "DEA_Message", "DEA_SuperEfficiency_Status"] if c in results_df]
    if solver_cols:
        with st.expander("**DEA solver diagnostics**"):
            st.dataframe(results_df[solver_cols], width="content")


def render_export_tab(
    results_df: pd.DataFrame,
    target_df: pd.DataFrame,
    peer_weights: pd.DataFrame,
    support: pd.DataFrame,
    sfa_param_df: pd.DataFrame,
    jackknife_df: Optional[pd.DataFrame],
    cross_eff: Optional[Dict[str, object]],
    dea_bootstrap: Optional[Dict[str, object]],
    sfa_bootstrap: Optional[Dict[str, object]],
    dmu_ids: Sequence[str],
) -> None:
    st.subheader("Download Results")

    csv_results = results_df.to_csv(index=False).encode("utf-8")
    display_download_button("Download full results CSV", csv_results, "efficiency_results.csv", "text/csv")

    csv_targets = target_df.to_csv(index=False).encode("utf-8")
    display_download_button("Download DEA targets CSV", csv_targets, "dea_targets.csv", "text/csv")

    csv_peers = peer_weights.to_csv(index=False).encode("utf-8")
    display_download_button("Download peer weights CSV", csv_peers, "dea_peer_weights.csv", "text/csv")

    if dea_bootstrap is not None and isinstance(dea_bootstrap.get("table"), pd.DataFrame):
        dea_boot_csv = dea_bootstrap["table"].to_csv(index=False).encode("utf-8")
        display_download_button("Download DEA bootstrap CSV", dea_boot_csv, "dea_bootstrap_intervals.csv", "text/csv")

    if sfa_bootstrap is not None and isinstance(sfa_bootstrap.get("efficiency_table"), pd.DataFrame):
        sfa_boot_csv = sfa_bootstrap["efficiency_table"].to_csv(index=False).encode("utf-8")
        display_download_button("Download SFA efficiency bootstrap CSV", sfa_boot_csv, "sfa_efficiency_bootstrap_intervals.csv", "text/csv")

    tables: Dict[str, pd.DataFrame] = {
        "Results": results_df,
        "DEA_Targets": target_df,
        "Peer_Weights": peer_weights,
        "Benchmark_Support": support,
        "SFA_Parameters": sfa_param_df,
    }
    if jackknife_df is not None:
        tables["Jackknife"] = jackknife_df
    if dea_bootstrap is not None:
        dea_boot_table = dea_bootstrap.get("table")
        if isinstance(dea_boot_table, pd.DataFrame):
            tables["DEA_Bootstrap"] = dea_boot_table
        dea_boot_scores = dea_bootstrap.get("score_matrix")
        if isinstance(dea_boot_scores, np.ndarray):
            tables["DEA_Boot_Scores"] = pd.DataFrame(dea_boot_scores, columns=dmu_ids)
    if sfa_bootstrap is not None:
        sfa_boot_params = sfa_bootstrap.get("parameter_table")
        sfa_boot_metrics = sfa_bootstrap.get("metric_table")
        sfa_boot_eff = sfa_bootstrap.get("efficiency_table")
        if isinstance(sfa_boot_params, pd.DataFrame):
            tables["SFA_Boot_Params"] = sfa_boot_params
        if isinstance(sfa_boot_metrics, pd.DataFrame):
            tables["SFA_Boot_Metrics"] = sfa_boot_metrics
        if isinstance(sfa_boot_eff, pd.DataFrame):
            tables["SFA_Boot_Eff"] = sfa_boot_eff
    if cross_eff is not None:
        tables["Cross_Eff_Matrix"] = pd.DataFrame(cross_eff["matrix"], index=dmu_ids, columns=dmu_ids).reset_index(names="Evaluator_DMU_ID")

    try:
        workbook = make_excel_download(tables)
        display_download_button(
            "Download Excel workbook",
            workbook,
            "efficiency_analysis_workbook.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as exc:
        st.warning(f"Excel workbook export failed: {exc}. CSV exports are still available.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("Advanced DEA & SFA Analysis")

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose an Excel or CSV file", type=["xlsx", "csv"])

    if uploaded_file is None:
        render_intro()
        return

    try:
        df = load_uploaded_data(uploaded_file.name, uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read uploaded file: {exc}")
        return

    st.sidebar.success("File uploaded successfully.")
    with st.sidebar.expander("Data preview"):
        st.write(df.head())
        st.write(f"Shape: {df.shape}")

    config = render_sidebar(df)
    input_cols = list(config["input_cols"])
    output_cols = list(config["output_cols"])

    if not input_cols or not output_cols:
        st.info("Select input and output columns in the sidebar to run DEA and SFA.")
        return

    df_clean, summary = clean_and_prepare_data(
        df=df,
        input_cols=input_cols,
        output_cols=output_cols,
        id_col=config["id_col"],
        time_col=config["time_col"],
        group_col=config["group_col"],
        env_cols=config["env_cols"],
        selected_groups=config["selected_groups"],
    )

    errors, validation_warnings = validate_analysis_setup(
        df_clean=df_clean,
        input_cols=input_cols,
        output_cols=output_cols,
        id_col=config["id_col"],
        sfa_model=config["sfa_model"],
    )

    render_validation(summary, validation_warnings)
    if errors:
        for error in errors:
            st.error(error)
        return

    if len(output_cols) > 1:
        st.info(
            f"DEA will use all {len(output_cols)} selected outputs. SFA will use the selected single output: {config['sfa_output_col']}."
        )

    dmu_ids = df_clean["DMU_ID"].astype(str).tolist()
    metadata_cols = unique_keep_order([config["id_col"], config["time_col"], config["group_col"]] + list(config["env_cols"]))

    with st.spinner("Computing DEA, SFA, scale diagnostics, and selected advanced modules..."):
        dea = calculate_dea_with_slacks_cached(
            df_clean,
            tuple(input_cols),
            tuple(output_cols),
            config["dea_orientation"],
            config["dea_returns"],
        )
        scale = calculate_scale_diagnostics_cached(
            df_clean,
            tuple(input_cols),
            tuple(output_cols),
            config["dea_orientation"],
        )
        support = calculate_reference_support(np.asarray(dea["lambdas"]), np.asarray(dea["efficiency"]), dmu_ids)

        super_eff: Optional[Dict[str, object]] = None
        if config["run_super_eff"]:
            super_eff = calculate_super_efficiency_cached(
                df_clean,
                tuple(input_cols),
                tuple(output_cols),
                config["dea_orientation"],
                config["dea_returns"],
            )

        sfa = calculate_sfa_production_cached(
            df_clean,
            tuple(input_cols),
            str(config["sfa_output_col"]),
            config["sfa_model"],
            config["sfa_frontier_type"],
        )

        cross_eff: Optional[Dict[str, object]] = None
        if config["run_cross_eff"]:
            cross_eff = calculate_cross_efficiency_crs_cached(df_clean, tuple(input_cols), tuple(output_cols))

        jackknife_df: Optional[pd.DataFrame] = None
        if config["run_jackknife"]:
            if len(df_clean) <= 80:
                jackknife_df = calculate_jackknife_influence_cached(
                    df_clean,
                    tuple(input_cols),
                    tuple(output_cols),
                    config["dea_orientation"],
                    config["dea_returns"],
                )
            else:
                st.warning("Leave-one-out robustness was skipped because more than 80 DMUs were selected.")

        dea_bootstrap: Optional[Dict[str, object]] = None
        if config["run_dea_bootstrap"]:
            requested_lps = len(df_clean) * int(config["dea_bootstrap_replications"])
            if requested_lps <= 30000:
                dea_bootstrap = calculate_dea_bootstrap_cached(
                    df_clean,
                    tuple(input_cols),
                    tuple(output_cols),
                    config["dea_orientation"],
                    config["dea_returns"],
                    int(config["dea_bootstrap_replications"]),
                    float(config["bootstrap_ci_level"]),
                    int(config["bootstrap_seed"]),
                    bool(config["dea_bootstrap_include_self"]),
                )
            else:
                st.warning(
                    "DEA bootstrap was skipped because the selected settings require "
                    f"{requested_lps:,} LP solves. Reduce DMUs or replications below 30,000 solves."
                )

        sfa_bootstrap: Optional[Dict[str, object]] = None
        if config["run_sfa_bootstrap"]:
            if len(df_clean) <= 500:
                sfa_bootstrap = calculate_sfa_bootstrap_cached(
                    df_clean,
                    tuple(input_cols),
                    str(config["sfa_output_col"]),
                    config["sfa_model"],
                    config["sfa_frontier_type"],
                    int(config["sfa_bootstrap_replications"]),
                    float(config["bootstrap_ci_level"]),
                    int(config["bootstrap_seed"]),
                )
            else:
                st.warning("SFA bootstrap was skipped because more than 500 DMUs were selected.")

    target_df = build_target_table(df_clean, dea, input_cols, output_cols, dmu_ids)
    peer_weights = build_peer_weights_table(np.asarray(dea["lambdas"]), dmu_ids)
    sfa_param_df = sfa_parameters_table(sfa)
    results_df = build_results_table(
        df_clean=df_clean,
        dea=dea,
        scale=scale,
        sfa=sfa,
        super_eff=super_eff,
        support=support,
        cross_eff=cross_eff,
        metadata_cols=metadata_cols,
        input_cols=input_cols,
        output_cols=output_cols,
    )

    if dea_bootstrap is not None and isinstance(dea_bootstrap.get("table"), pd.DataFrame):
        dea_boot_cols = [
            c
            for c in dea_bootstrap["table"].columns
            if c == "DMU_ID" or c.startswith("DEA_Bootstrap_")
        ]
        results_df = results_df.merge(dea_bootstrap["table"][dea_boot_cols], on="DMU_ID", how="left")

    if sfa_bootstrap is not None and isinstance(sfa_bootstrap.get("efficiency_table"), pd.DataFrame):
        sfa_boot_cols = [
            c
            for c in sfa_bootstrap["efficiency_table"].columns
            if c == "DMU_ID" or c.startswith("SFA_Boot_")
        ]
        results_df = results_df.merge(sfa_bootstrap["efficiency_table"][sfa_boot_cols], on="DMU_ID", how="left")

    tabs = st.tabs(
        [
            "Results",
            "Targets & Peers",
            "DEA vs SFA",
            "Distributions",
            "Frontier",
            "Scale & RTS",
            "Benchmarks",
            "Super & Robustness",
            "SFA Details",
            "Diagnostics",
            "Export",
        ]
    )

    with tabs[0]:
        render_results_tab(results_df, results_df["DEA_Efficiency"].to_numpy(dtype=float), results_df["SFA_Efficiency"].to_numpy(dtype=float))
    with tabs[1]:
        render_targets_tab(target_df, peer_weights, dmu_ids)
    with tabs[2]:
        render_comparison_tab(results_df)
    with tabs[3]:
        render_distributions_tab(results_df)
    with tabs[4]:
        render_frontier_tab(df_clean, input_cols, output_cols, results_df)
    with tabs[5]:
        render_scale_tab(results_df)
    with tabs[6]:
        render_benchmark_tab(support, cross_eff, dmu_ids)
    with tabs[7]:
        render_super_and_robustness_tab(
            results_df,
            jackknife_df,
            dea_bootstrap,
            sfa_bootstrap,
            config["dea_orientation"],
            config["dea_returns"],
        )
    with tabs[8]:
        render_sfa_tab(sfa, sfa_param_df)
    with tabs[9]:
        render_diagnostics_tab(summary, errors, validation_warnings, results_df)
    with tabs[10]:
        render_export_tab(
            results_df,
            target_df,
            peer_weights,
            support,
            sfa_param_df,
            jackknife_df,
            cross_eff,
            dea_bootstrap,
            sfa_bootstrap,
            dmu_ids,
        )


if __name__ == "__main__":
    main()
