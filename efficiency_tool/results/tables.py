from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from efficiency_tool.config import PEER_TOL
from efficiency_tool.dea.core import peer_string
from efficiency_tool.sfa.core import COST_EFFICIENCY_STATA
from efficiency_tool.utils.data import unique_keep_order
from efficiency_tool.utils.stats import safe_rank


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


def _benchmarking_style_dea_score(efficiency: np.ndarray, radial_factor: np.ndarray, orientation: str) -> np.ndarray:
    """Return a score comparable to R Benchmarking::dea() eff values."""
    if str(orientation).lower() == "output":
        return radial_factor
    return efficiency


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
    dea_orientation: str = "output",
    sfa_input_cols: Optional[Sequence[str]] = None,
    sfa_output_col: Optional[str] = None,
) -> pd.DataFrame:
    dmu_ids = df_clean["DMU_ID"].astype(str).tolist()
    keep_cols = unique_keep_order(
        ["DMU_ID", "Original_Row"]
        + list(metadata_cols)
        + list(input_cols)
        + list(output_cols)
        + list(sfa_input_cols or [])
        + ([sfa_output_col] if sfa_output_col else [])
    )
    keep_cols = [col for col in keep_cols if col in df_clean.columns]
    results = df_clean.loc[:, keep_cols].copy()

    dea_eff = np.asarray(dea["efficiency"], dtype=float)
    dea_radial = np.asarray(dea["radial_factor"], dtype=float)
    results["DEA_Efficiency"] = dea_eff
    results["DEA_Efficiency_0_1"] = dea_eff
    results["DEA_Radial_Factor"] = dea_radial
    results["DEA_R_Benchmarking_Style_Score"] = _benchmarking_style_dea_score(dea_eff, dea_radial, dea_orientation)
    results["DEA_Rank"] = safe_rank(results["DEA_Efficiency"], ascending=False)
    if str(dea_orientation).lower() == "output":
        results["DEA_R_Benchmarking_Style_Rank"] = safe_rank(results["DEA_R_Benchmarking_Style_Score"], ascending=True)
    else:
        results["DEA_R_Benchmarking_Style_Rank"] = safe_rank(results["DEA_R_Benchmarking_Style_Score"], ascending=False)
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

    sfa_eff = np.asarray(sfa["efficiency"], dtype=float)
    results["SFA_Efficiency"] = sfa_eff
    results["SFA_Model_Input_Columns"] = ", ".join(map(str, sfa.get("input_cols", list(sfa_input_cols or []))))
    results["SFA_Model_Output_Column"] = str(sfa.get("output_col", sfa_output_col or ""))
    results["SFA_Variables_Already_Logged"] = bool(sfa.get("data_are_logged", False))
    results["SFA_Cost_Efficiency_Convention"] = str(sfa.get("cost_efficiency_convention", ""))
    sfa_lower_is_better = (
        str(sfa.get("frontier_type", "")).strip().lower() == "cost"
        and str(sfa.get("cost_efficiency_convention", "")) == COST_EFFICIENCY_STATA
    )
    results["SFA_Rank"] = safe_rank(results["SFA_Efficiency"], ascending=sfa_lower_is_better)

    support_for_merge = support[["DMU_ID", "Reference_Count", "Reference_Weight_Sum", "Benchmark_Support_Score"]]
    results = results.merge(support_for_merge, on="DMU_ID", how="left")

    if cross_eff is not None:
        results["CCR_Cross_Efficiency_Mean"] = np.asarray(cross_eff["mean_cross"], dtype=float)
        results["CCR_Cross_Efficiency_Min"] = np.asarray(cross_eff["min_cross"], dtype=float)
        results["CCR_Cross_Efficiency_Max"] = np.asarray(cross_eff["max_cross"], dtype=float)
        results["CCR_Self_Efficiency_from_Multiplier"] = np.asarray(cross_eff["self_cross"], dtype=float)
        results["Self_vs_Cross_Gap"] = results["DEA_Efficiency"] - results["CCR_Cross_Efficiency_Mean"]

    return results
