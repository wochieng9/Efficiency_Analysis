from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from efficiency_tool.sfa.core import COST_EFFICIENCY_01, COST_EFFICIENCY_STATA
from efficiency_tool.ui.widgets import add_sidebar_selectbox
from efficiency_tool.utils.data import likely_numeric_columns


def _index_or_zero(options: List[str], preferred: Optional[str]) -> int:
    if preferred in options:
        return options.index(preferred)  # type: ignore[arg-type]
    return 0


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
    st.sidebar.subheader("DEA specification")
    input_cols = st.sidebar.multiselect(
        "DEA input columns",
        options=numeric_candidates,
        help="Resources or costs to minimize, such as labor, beds, expenditure, or capital. DEA uses positive level variables.",
    )
    output_cols = st.sidebar.multiselect(
        "DEA output columns",
        options=numeric_candidates,
        help="Services or outcomes to maximize. DEA can use multiple positive level outputs.",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        dea_orientation = st.selectbox(
            "DEA orientation",
            ["input", "output"],
            help="R Benchmarking::dea() defaults to input orientation.",
        )
    with col2:
        dea_returns = st.selectbox(
            "DEA returns",
            ["vrs", "crs"],
            help="R Benchmarking::dea() defaults to VRS.",
        )

    dea_time_mode = "Pooled DMU-period frontier"
    if time_col is not None:
        dea_time_mode = st.sidebar.selectbox(
            "DEA time handling",
            ["Pooled DMU-period frontier", "Separate frontier within each time period"],
            help=(
                "Use pooled when every DMU-period should share one technology. Use separate periods when matching year-by-year DEA runs. "
                "Malmquist below always uses adjacent-period reference technologies."
            ),
        )

    run_malmquist = False
    malmquist_same_reference = False
    if id_col is not None and time_col is not None:
        run_malmquist = st.sidebar.checkbox(
            "Run Malmquist productivity index",
            value=True,
            help=(
                "Computes adjacent-period DEA Malmquist MPI, efficiency change, and technical change for units present in both periods."
            ),
        )
        malmquist_same_reference = st.sidebar.checkbox(
            "Malmquist: use only common units as reference",
            value=False,
            help=(
                "Unchecked matches the usual R Benchmarking default: each period frontier uses all available units in that period. "
                "Checked restricts both adjacent frontiers to units present in both periods."
            ),
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("SFA specification")
    sfa_default_inputs = [col for col in input_cols if col in numeric_candidates]
    sfa_input_cols = st.sidebar.multiselect(
        "SFA input/regressor columns",
        options=numeric_candidates,
        default=sfa_default_inputs,
        help=(
            "These can be the same raw level variables used in DEA, or already logged variables if you check the logged-variable option below."
        ),
    )

    preferred_sfa_output = output_cols[0] if output_cols else (numeric_candidates[0] if numeric_candidates else None)
    if numeric_candidates:
        sfa_output_col = st.sidebar.selectbox(
            "SFA dependent/output column",
            options=numeric_candidates,
            index=_index_or_zero(numeric_candidates, preferred_sfa_output),
            help="SFA in this app is single-output. Select the same dependent variable you use in R/Stata.",
        )
    else:
        sfa_output_col = None

    sfa_data_are_logged = st.sidebar.checkbox(
        "SFA selected variables are already logged",
        value=False,
        help=(
            "Leave unchecked when selecting raw positive variables; the app will take natural logs internally. "
            "Check this only when the selected SFA dependent and regressor columns are already log-transformed."
        ),
    )
    sfa_model = st.sidebar.selectbox("SFA frontier form", ["Cobb-Douglas", "Translog"])
    sfa_frontier_type = st.sidebar.selectbox("SFA frontier type", ["Production", "Cost"])
    sfa_cost_efficiency_convention = st.sidebar.selectbox(
        "Cost-frontier efficiency convention",
        [COST_EFFICIENCY_01, COST_EFFICIENCY_STATA],
        help=(
            "For cost frontiers, some packages report a cost ratio greater than or equal to 1. "
            "Use the Stata/FRONTIER option only when matching that convention."
        ),
    )

    env_options = [col for col in numeric_candidates if col not in set(input_cols).union(output_cols).union(sfa_input_cols).union({sfa_output_col})]
    env_cols = st.sidebar.multiselect(
        "Environmental / case-mix columns to preserve",
        options=env_options,
        help="Optional columns kept in exports for stratification or second-stage analysis. They are not treated as discretionary DEA inputs/outputs.",
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
        "dea_time_mode": dea_time_mode,
        "run_malmquist": bool(run_malmquist),
        "malmquist_same_reference": bool(malmquist_same_reference),
        "sfa_input_cols": sfa_input_cols,
        "sfa_output_col": sfa_output_col,
        "sfa_data_are_logged": bool(sfa_data_are_logged),
        "sfa_model": sfa_model,
        "sfa_frontier_type": sfa_frontier_type,
        "sfa_cost_efficiency_convention": sfa_cost_efficiency_convention,
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
