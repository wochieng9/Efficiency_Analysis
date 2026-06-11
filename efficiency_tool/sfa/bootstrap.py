from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from efficiency_tool.sfa.core import COST_EFFICIENCY_STATA, calculate_sfa_efficiency_from_fit, calculate_sfa_production
from efficiency_tool.utils.stats import column_mean, column_percentile, column_std

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
    data_are_logged: bool = False,
    cost_efficiency_convention: str = "0-1 reciprocal E[exp(-u)|eps]",
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

    baseline_fit = calculate_sfa_production(
        data,
        input_cols,
        output_col,
        model_type,
        frontier_type,
        data_are_logged=data_are_logged,
        cost_efficiency_convention=cost_efficiency_convention,
    )
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
        fit = calculate_sfa_production(
            boot_data,
            input_cols,
            output_col,
            model_type,
            frontier_type,
            data_are_logged=data_are_logged,
            cost_efficiency_convention=cost_efficiency_convention,
        )
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
            data_are_logged=data_are_logged,
            cost_efficiency_convention=cost_efficiency_convention,
        )
        usable += 1
        if bool(fit.get("converged", False)):
            converged += 1

    eff_mean = column_mean(eff_scores)
    eff_std = column_std(eff_scores, ddof=1)
    eff_bias = eff_mean - baseline_eff
    valid_eff = np.sum(np.isfinite(eff_scores), axis=0)
    if cost_efficiency_convention == COST_EFFICIENCY_STATA:
        bias_corrected = np.maximum(baseline_eff - eff_bias, 1.0)
    else:
        bias_corrected = np.clip(baseline_eff - eff_bias, 0.0, 1.0)

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
            "SFA_Boot_Bias_Corrected": bias_corrected,
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
        "data_are_logged": bool(data_are_logged),
        "cost_efficiency_convention": cost_efficiency_convention,
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

