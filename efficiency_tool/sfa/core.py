from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from efficiency_tool.config import TOL


COST_EFFICIENCY_01 = "0-1 reciprocal E[exp(-u)|eps]"
COST_EFFICIENCY_STATA = "Stata/FRONTIER cost ratio E[exp(u)|eps]"


def build_sfa_design(
    log_X: np.ndarray,
    input_cols: Sequence[str],
    model_type: str,
    data_are_logged: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Build Cobb-Douglas or Translog design matrix using logged inputs."""
    n_units, n_inputs = log_X.shape
    cols = [np.ones(n_units)]
    names = ["Intercept"]

    def label(col: str) -> str:
        return str(col) if data_are_logged else f"ln({col})"

    for i, col in enumerate(input_cols):
        cols.append(log_X[:, i])
        names.append(label(str(col)))

    if model_type == "Translog":
        for i, col in enumerate(input_cols):
            cols.append(0.5 * log_X[:, i] ** 2)
            names.append(f"0.5*{label(str(col))}^2")
        for i in range(n_inputs):
            for j in range(i + 1, n_inputs):
                cols.append(log_X[:, i] * log_X[:, j])
                names.append(f"{label(str(input_cols[i]))}*{label(str(input_cols[j]))}")

    return np.column_stack(cols), names


def prepare_sfa_logs(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: str,
    data_are_logged: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return logged design values and logged dependent variable for SFA."""
    X_raw = data[list(input_cols)].to_numpy(dtype=float)
    y_raw = data[output_col].to_numpy(dtype=float)

    if data_are_logged:
        if not np.all(np.isfinite(X_raw)) or not np.all(np.isfinite(y_raw)):
            return X_raw, y_raw, "SFA logged variables must be finite numeric values."
        return X_raw, y_raw, ""

    if np.any(X_raw <= 0) or np.any(y_raw <= 0):
        return X_raw, y_raw, "SFA requires strictly positive selected inputs and output when variables are not already logged."
    return np.log(X_raw), np.log(y_raw), ""


def conditional_efficiency_from_residuals(
    eps: np.ndarray,
    sigma_u: float,
    sigma_v: float,
    frontier_type: str,
    cost_efficiency_convention: str = COST_EFFICIENCY_01,
) -> np.ndarray:
    """Compute conditional efficiency from half-normal SFA residuals.

    Production frontier: ln(y) = f(x) + v - u.
    Cost frontier: ln(c) = f(x) + v + u.

    By default cost efficiency is reported as E[exp(-u)|eps], bounded by 0 and 1.
    The Stata/FRONTIER convention for cost frontiers reports E[exp(u)|eps], where
    1 is best and values above 1 indicate excess cost.
    """
    eps_arr = np.asarray(eps, dtype=float)
    if not np.isfinite(sigma_u) or not np.isfinite(sigma_v) or sigma_u <= 0 or sigma_v <= 0:
        return np.full_like(eps_arr, np.nan, dtype=float)

    sigma_sq = float(sigma_u) ** 2 + float(sigma_v) ** 2
    if sigma_sq <= 0:
        return np.full_like(eps_arr, np.nan, dtype=float)

    frontier_type_norm = str(frontier_type).strip().lower()
    q = 1.0 if frontier_type_norm == "production" else -1.0
    mu_star = -q * eps_arr * float(sigma_u) ** 2 / sigma_sq
    sigma_star_sq = (float(sigma_u) ** 2 * float(sigma_v) ** 2) / sigma_sq
    sigma_star = math.sqrt(max(sigma_star_sq, TOL))
    denom_arg = mu_star / sigma_star

    cost_stata = frontier_type_norm == "cost" and str(cost_efficiency_convention) == COST_EFFICIENCY_STATA
    if cost_stata:
        # E[exp(u)|eps]
        log_eff = (
            mu_star
            + 0.5 * sigma_star_sq
            + norm.logcdf(denom_arg + sigma_star)
            - norm.logcdf(denom_arg)
        )
        eff = np.exp(np.clip(log_eff, -745, 709))
        eff = np.where(np.isfinite(eff), np.maximum(eff, 1.0), np.nan)
        return eff

    # E[exp(-u)|eps]
    log_eff = (
        -mu_star
        + 0.5 * sigma_star_sq
        + norm.logcdf(denom_arg - sigma_star)
        - norm.logcdf(denom_arg)
    )
    eff = np.exp(np.clip(log_eff, -745, 0))
    return np.where(np.isfinite(eff), np.clip(eff, 0.0, 1.0), np.nan)


def calculate_sfa_production(
    data: pd.DataFrame,
    input_cols: Sequence[str],
    output_col: str,
    model_type: str = "Cobb-Douglas",
    frontier_type: str = "Production",
    data_are_logged: bool = False,
    cost_efficiency_convention: str = COST_EFFICIENCY_01,
) -> Dict[str, object]:
    """Half-normal SFA estimated by maximum likelihood.

    Production frontier: log(y) = f(x) + v - u.
    Cost frontier: log(c) = f(x) + v + u.
    Efficiency is reported according to ``cost_efficiency_convention`` for cost
    frontiers and as E[exp(-u)|epsilon] for production frontiers.
    """
    n_units = len(data)

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
        "input_cols": list(input_cols),
        "data_are_logged": bool(data_are_logged),
        "cost_efficiency_convention": cost_efficiency_convention,
    }

    if n_units < 5:
        empty_result["message"] = "SFA needs more observations for reliable estimation."
        return empty_result

    log_X, log_y, prep_error = prepare_sfa_logs(data, input_cols, output_col, data_are_logged)
    if prep_error:
        empty_result["message"] = prep_error
        return empty_result

    X_reg, param_names = build_sfa_design(log_X, input_cols, model_type, data_are_logged=data_are_logged)
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
    lam = sigma_u / sigma_v
    gamma = sigma_u**2 / sigma_sq
    eps = log_y - X_reg @ beta

    te = conditional_efficiency_from_residuals(
        eps,
        sigma_u,
        sigma_v,
        frontier_type,
        cost_efficiency_convention=cost_efficiency_convention,
    )

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
        "input_cols": list(input_cols),
        "data_are_logged": bool(data_are_logged),
        "cost_efficiency_convention": cost_efficiency_convention,
        "log_residuals": eps,
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
    data_are_logged: bool = False,
    cost_efficiency_convention: str = COST_EFFICIENCY_01,
) -> np.ndarray:
    """Evaluate original observations using fitted half-normal SFA parameters."""
    n_units = len(data)
    beta_arr = np.asarray(beta, dtype=float)
    if len(beta_arr) == 0 or not np.isfinite(sigma_u) or not np.isfinite(sigma_v) or sigma_u <= 0 or sigma_v <= 0:
        return np.full(n_units, np.nan)

    log_X, log_y, prep_error = prepare_sfa_logs(data, input_cols, output_col, data_are_logged)
    if prep_error:
        return np.full(n_units, np.nan)

    X_reg, _ = build_sfa_design(log_X, input_cols, model_type, data_are_logged=data_are_logged)
    if X_reg.shape[1] != len(beta_arr):
        return np.full(n_units, np.nan)

    eps = log_y - X_reg @ beta_arr
    return conditional_efficiency_from_residuals(
        eps,
        sigma_u,
        sigma_v,
        frontier_type,
        cost_efficiency_convention=cost_efficiency_convention,
    )
