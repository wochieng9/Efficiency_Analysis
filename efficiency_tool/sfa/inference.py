from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _is_first_order_param(name: str) -> bool:
    return name != "Intercept" and "*" not in name and "^" not in name and not name.startswith("0.5*")


def sfa_coefficient_table_with_inference(sfa: dict) -> pd.DataFrame:
    """
    Build a publication-ready SFA coefficient table with coefficients,
    approximate standard errors, test statistics, confidence intervals, and
    Cobb-Douglas elasticity interpretation.
    """
    param_names = list(sfa.get("param_names", []))
    beta = np.asarray(sfa.get("beta", []), dtype=float)
    se = np.asarray(sfa.get("std_errors", []), dtype=float)
    model_type = str(sfa.get("model_type", "Cobb-Douglas"))

    if not param_names or len(beta) == 0:
        return pd.DataFrame(columns=[
            "Parameter",
            "Coefficient",
            "Std_Error",
            "t_stat",
            "p_value",
            "CI_Lower_95%",
            "CI_Upper_95%",
            "Elasticity",
        ])

    if len(beta) != len(param_names):
        beta = np.full(len(param_names), np.nan)
    if len(se) != len(param_names):
        se = np.full(len(param_names), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
        p_values = np.where(np.isfinite(t_stats), 2 * (1 - norm.cdf(np.abs(t_stats))), np.nan)
        ci_lower = beta - 1.96 * se
        ci_upper = beta + 1.96 * se

    elasticity = np.full_like(beta, np.nan)
    if model_type == "Cobb-Douglas":
        for i, name in enumerate(param_names):
            if _is_first_order_param(str(name)):
                elasticity[i] = beta[i]

    return pd.DataFrame({
        "Parameter": param_names,
        "Coefficient": beta,
        "Std_Error": se,
        "t_stat": t_stats,
        "p_value": p_values,
        "CI_Lower_95%": ci_lower,
        "CI_Upper_95%": ci_upper,
        "Elasticity": elasticity,
    })


def sfa_equation_latex(
    sfa: dict,
    output_col: str,
    frontier_type: str = "Production",
) -> str:
    """Render the estimated SFA equation in readable form."""
    param_names = list(sfa.get("param_names", []))
    beta = np.asarray(sfa.get("beta", []), dtype=float)
    data_are_logged = bool(sfa.get("data_are_logged", False))
    if not param_names or len(beta) == 0:
        return "SFA equation not available (model did not estimate)."

    frontier_norm = str(frontier_type).strip().lower()
    frontier_side = "+ v - u" if frontier_norm == "production" else "+ v + u"

    rhs_terms = []
    for i, name in enumerate(param_names):
        coeff = beta[i]
        if np.isnan(coeff):
            continue

        sign = "+" if coeff >= 0 else ""
        name_str = str(name)
        if name_str.startswith("Intercept"):
            rhs_terms.append(f"{sign} {coeff:.6g}")
        elif _is_first_order_param(name_str):
            rhs_terms.append(f"{sign} {coeff:.4f}·{name_str}")
        elif "^2" in name_str or name_str.startswith("0.5*"):
            var = name_str.split("^")[0].replace("0.5*", "")
            rhs_terms.append(f"{sign} {coeff:.4f}·({var})²")
        elif "*" in name_str:
            vars_in = name_str.split("*")
            rhs_terms.append(f"{sign} {coeff:.4f}·" + "·".join(vars_in))

    rhs = " ".join(rhs_terms) + f" {frontier_side}"
    lhs = str(output_col) if data_are_logged else f"ln({output_col})"
    return f"{lhs} = " + rhs


def sfa_returns_to_scale_test(sfa: dict) -> dict:
    """Approximate CRS test based on the sum of first-order Cobb-Douglas elasticities."""
    param_names = list(sfa.get("param_names", []))
    beta = np.asarray(sfa.get("beta", []), dtype=float)
    se = np.asarray(sfa.get("std_errors", []), dtype=float)

    if not param_names or len(beta) == 0:
        return {
            "test": "unavailable",
            "sum_elasticity": np.nan,
            "se_sum": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conclusion": "Cannot perform test: model not estimated.",
        }

    first_order_indices = [i for i, name in enumerate(param_names) if _is_first_order_param(str(name))]

    if not first_order_indices:
        return {
            "test": "unavailable",
            "sum_elasticity": np.nan,
            "se_sum": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conclusion": "No first-order input terms found.",
        }

    sum_beta = float(np.sum(beta[first_order_indices]))
    se_sum = float(np.sqrt(np.sum(se[first_order_indices] ** 2)))

    if se_sum > 0:
        t_stat = (sum_beta - 1.0) / se_sum
        p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
    else:
        t_stat = np.nan
        p_value = np.nan

    if np.isfinite(p_value) and p_value < 0.05:
        if sum_beta > 1:
            conclusion = f"REJECT CRS (p={p_value:.4f}): Increasing returns to scale (Σ elasticity = {sum_beta:.4f})"
        else:
            conclusion = f"REJECT CRS (p={p_value:.4f}): Decreasing returns to scale (Σ elasticity = {sum_beta:.4f})"
    elif np.isfinite(p_value):
        conclusion = f"FAIL TO REJECT CRS (p={p_value:.4f}): Data consistent with constant returns (Σ elasticity = {sum_beta:.4f})"
    else:
        conclusion = "Cannot perform test (insufficient standard error data)."

    return {
        "test": "Constant Returns to Scale",
        "sum_elasticity": sum_beta,
        "se_sum": se_sum,
        "t_stat": t_stat,
        "p_value": p_value,
        "conclusion": conclusion,
        "num_inputs": len(first_order_indices),
    }
