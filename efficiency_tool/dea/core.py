from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from efficiency_tool.config import PEER_TOL, TOL

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



def _assign_dea_subresult(
    combined: Dict[str, object],
    subresult: Dict[str, object],
    idx: np.ndarray,
) -> None:
    """Place one group-level DEA result back into global row order."""
    combined["efficiency"][idx] = np.asarray(subresult["efficiency"], dtype=float)
    combined["radial_factor"][idx] = np.asarray(subresult["radial_factor"], dtype=float)
    combined["input_slacks"][idx, :] = np.asarray(subresult["input_slacks"], dtype=float)
    combined["output_slacks"][idx, :] = np.asarray(subresult["output_slacks"], dtype=float)
    combined["radial_input_reduction"][idx, :] = np.asarray(subresult["radial_input_reduction"], dtype=float)
    combined["radial_output_increase"][idx, :] = np.asarray(subresult["radial_output_increase"], dtype=float)
    combined["target_inputs"][idx, :] = np.asarray(subresult["target_inputs"], dtype=float)
    combined["target_outputs"][idx, :] = np.asarray(subresult["target_outputs"], dtype=float)
    combined["lambdas"][np.ix_(idx, idx)] = np.asarray(subresult["lambdas"], dtype=float)
    statuses = list(subresult["statuses"])
    messages = list(subresult["messages"])
    for pos, row_idx in enumerate(idx):
        combined["statuses"][int(row_idx)] = statuses[pos]
        combined["messages"][int(row_idx)] = messages[pos]


def calculate_dea_with_slacks_by_group(
    data: pd.DataFrame,
    group_col: str,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    """Run DEA separately within each level of a grouping column, preserving global row order."""
    if group_col not in data.columns:
        return calculate_dea_with_slacks(data, input_cols, output_cols, orientation, returns)

    n_units = len(data)
    n_inputs = len(input_cols)
    n_outputs = len(output_cols)
    combined: Dict[str, object] = {
        "efficiency": np.full(n_units, np.nan),
        "radial_factor": np.full(n_units, np.nan),
        "input_slacks": np.full((n_units, n_inputs), np.nan),
        "output_slacks": np.full((n_units, n_outputs), np.nan),
        "radial_input_reduction": np.zeros((n_units, n_inputs)),
        "radial_output_increase": np.zeros((n_units, n_outputs)),
        "target_inputs": np.full((n_units, n_inputs), np.nan),
        "target_outputs": np.full((n_units, n_outputs), np.nan),
        "lambdas": np.zeros((n_units, n_units)),
        "statuses": ["not_run"] * n_units,
        "messages": [""] * n_units,
        "reference_group_col": group_col,
    }

    for _, sub in data.groupby(group_col, sort=False, dropna=False):
        idx = sub.index.to_numpy(dtype=int)
        subresult = calculate_dea_with_slacks(
            sub.reset_index(drop=True),
            input_cols,
            output_cols,
            orientation=orientation,
            returns=returns,
        )
        _assign_dea_subresult(combined, subresult, idx)

    return combined


def calculate_super_efficiency_by_group(
    data: pd.DataFrame,
    group_col: str,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str = "output",
    returns: str = "crs",
) -> Dict[str, object]:
    """Run super-efficiency separately within each group."""
    if group_col not in data.columns:
        return calculate_super_efficiency(data, input_cols, output_cols, orientation, returns)

    n_units = len(data)
    scores = np.full(n_units, np.nan)
    statuses: List[str] = ["not_run"] * n_units
    messages: List[str] = [""] * n_units

    for _, sub in data.groupby(group_col, sort=False, dropna=False):
        idx = sub.index.to_numpy(dtype=int)
        subresult = calculate_super_efficiency(
            sub.reset_index(drop=True),
            input_cols,
            output_cols,
            orientation=orientation,
            returns=returns,
        )
        scores[idx] = np.asarray(subresult["super_efficiency"], dtype=float)
        sub_statuses = list(subresult["statuses"])
        sub_messages = list(subresult["messages"])
        for pos, row_idx in enumerate(idx):
            statuses[int(row_idx)] = sub_statuses[pos]
            messages[int(row_idx)] = sub_messages[pos]

    return {"super_efficiency": scores, "statuses": statuses, "messages": messages, "reference_group_col": group_col}


def calculate_scale_diagnostics_by_group(
    data: pd.DataFrame,
    group_col: str,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    orientation: str,
) -> Dict[str, object]:
    """Compute CRS/VRS scale diagnostics separately within each group."""
    crs = calculate_dea_with_slacks_by_group(data, group_col, input_cols, output_cols, orientation, "crs")
    vrs = calculate_dea_with_slacks_by_group(data, group_col, input_cols, output_cols, orientation, "vrs")

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
        "reference_group_col": group_col,
    }
