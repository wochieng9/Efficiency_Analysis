from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from efficiency_tool.config import TOL
from efficiency_tool.dea.core import calculate_dea_with_slacks
from efficiency_tool.utils.stats import column_mean, column_percentile, column_std

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

