from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from efficiency_tool.app_cache import (
    calculate_cross_efficiency_crs_cached,
    calculate_dea_bootstrap_cached,
    calculate_dea_with_slacks_cached,
    calculate_jackknife_influence_cached,
    calculate_malmquist_indices_cached,
    calculate_scale_diagnostics_cached,
    calculate_sfa_bootstrap_cached,
    calculate_sfa_production_cached,
    calculate_super_efficiency_cached,
    load_uploaded_data,
)
from efficiency_tool.config import APP_TITLE
from efficiency_tool.dea.core import calculate_reference_support
from efficiency_tool.results.tables import build_peer_weights_table, build_results_table, build_target_table
from efficiency_tool.results.time_trends import build_time_trend_tables
from efficiency_tool.sfa.inference import sfa_coefficient_table_with_inference
from efficiency_tool.ui.sidebar import render_sidebar
from efficiency_tool.ui.sfa_details import render_sfa_tab_enhanced
from efficiency_tool.ui.tabs import (
    render_benchmark_tab,
    render_comparison_tab,
    render_diagnostics_tab,
    render_distributions_tab,
    render_export_tab,
    render_frontier_tab,
    render_intro,
    render_malmquist_tab,
    render_results_tab,
    render_scale_tab,
    render_super_and_robustness_tab,
    render_targets_tab,
    render_time_trends_tab,
    render_validation,
)
from efficiency_tool.utils.data import clean_and_prepare_data, unique_keep_order, validate_analysis_setup

org_icon = Image.open("aihp.png")
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(org_icon, width=240)
    with col2:
        st.title("DEA & SFA Analysis Tool")

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
    sfa_input_cols = list(config["sfa_input_cols"])
    sfa_output_col = config["sfa_output_col"]

    if not input_cols or not output_cols:
        st.info("Select DEA input and output columns in the sidebar to run the app.")
        return
    if not sfa_input_cols or not sfa_output_col:
        st.info("Select SFA input/regressor columns and one SFA dependent/output column in the sidebar.")
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
        sfa_input_cols=sfa_input_cols,
        sfa_output_col=str(sfa_output_col),
        sfa_data_are_logged=bool(config["sfa_data_are_logged"]),
    )

    errors, validation_warnings = validate_analysis_setup(
        df_clean=df_clean,
        input_cols=input_cols,
        output_cols=output_cols,
        id_col=config["id_col"],
        sfa_model=config["sfa_model"],
        sfa_input_cols=sfa_input_cols,
        sfa_output_col=str(sfa_output_col),
        sfa_data_are_logged=bool(config["sfa_data_are_logged"]),
        time_col=config["time_col"],
    )

    render_validation(summary, validation_warnings)
    if errors:
        for error in errors:
            st.error(error)
        return

    if len(output_cols) > 1:
        st.info(
            f"DEA will use all {len(output_cols)} selected outputs. SFA will use the selected single dependent/output column: {sfa_output_col}."
        )

    dea_reference_group_col: Optional[str] = None
    if config.get("time_col") is not None and str(config.get("dea_time_mode", "")).startswith("Separate"):
        dea_reference_group_col = str(config["time_col"])
        st.info(f"DEA is being estimated separately within each {dea_reference_group_col} period.")

    if str(config["dea_orientation"]).lower() == "output":
        st.caption(
            "For output-oriented DEA, DEA_Efficiency is reported as 1 / phi on a 0-1 scale. "
            "Use DEA_Radial_Factor or DEA_R_Benchmarking_Style_Score when comparing to packages that report the output expansion factor phi."
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
            dea_reference_group_col,
        )
        scale = calculate_scale_diagnostics_cached(
            df_clean,
            tuple(input_cols),
            tuple(output_cols),
            config["dea_orientation"],
            dea_reference_group_col,
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
                dea_reference_group_col,
            )

        sfa = calculate_sfa_production_cached(
            df_clean,
            tuple(sfa_input_cols),
            str(sfa_output_col),
            config["sfa_model"],
            config["sfa_frontier_type"],
            bool(config["sfa_data_are_logged"]),
            str(config["sfa_cost_efficiency_convention"]),
        )

        cross_eff: Optional[Dict[str, object]] = None
        if config["run_cross_eff"]:
            if dea_reference_group_col:
                st.warning("CCR cross-efficiency was skipped because DEA is being run separately by time period.")
            else:
                cross_eff = calculate_cross_efficiency_crs_cached(df_clean, tuple(input_cols), tuple(output_cols))

        jackknife_df: Optional[pd.DataFrame] = None
        if config["run_jackknife"]:
            if dea_reference_group_col:
                st.warning("Leave-one-out robustness was skipped because DEA is being run separately by time period.")
            elif len(df_clean) <= 80:
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
            if dea_reference_group_col:
                st.warning("DEA bootstrap was skipped because DEA is being run separately by time period.")
            else:
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
                    tuple(sfa_input_cols),
                    str(sfa_output_col),
                    config["sfa_model"],
                    config["sfa_frontier_type"],
                    int(config["sfa_bootstrap_replications"]),
                    float(config["bootstrap_ci_level"]),
                    int(config["bootstrap_seed"]),
                    bool(config["sfa_data_are_logged"]),
                    str(config["sfa_cost_efficiency_convention"]),
                )
            else:
                st.warning("SFA bootstrap was skipped because more than 500 DMUs were selected.")

        malmquist: Dict[str, object] = {"available": False, "reason": "Malmquist was not requested."}
        if bool(config.get("run_malmquist", False)):
            n_periods = df_clean[str(config["time_col"])].nunique(dropna=True) if config.get("time_col") is not None else 0
            max_malmquist_lps = 4 * len(df_clean) * max(int(n_periods) - 1, 0)
            if max_malmquist_lps <= 50000:
                malmquist = calculate_malmquist_indices_cached(
                    df_clean,
                    tuple(input_cols),
                    tuple(output_cols),
                    str(config["id_col"]) if config.get("id_col") is not None else None,
                    str(config["time_col"]) if config.get("time_col") is not None else None,
                    str(config["dea_orientation"]),
                    str(config["dea_returns"]),
                    bool(config.get("malmquist_same_reference", False)),
                )
            else:
                malmquist = {
                    "available": False,
                    "reason": (
                        "Malmquist was skipped because the selected panel could require up to "
                        f"{max_malmquist_lps:,} DEA LP solves. Filter the data or reduce periods."
                    ),
                }
                st.warning(str(malmquist["reason"]))

    target_df = build_target_table(df_clean, dea, input_cols, output_cols, dmu_ids)
    peer_weights = build_peer_weights_table(np.asarray(dea["lambdas"]), dmu_ids)
    sfa_param_df = sfa_coefficient_table_with_inference(sfa)
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
        dea_orientation=str(config["dea_orientation"]),
        sfa_input_cols=sfa_input_cols,
        sfa_output_col=str(sfa_output_col),
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

    time_trends = build_time_trend_tables(
        results_df=results_df,
        unit_col=str(config["id_col"]) if config.get("id_col") is not None else None,
        time_col=str(config["time_col"]) if config.get("time_col") is not None else None,
    )

    tab_names = [
        "Results",
        "Targets & Peers",
        "DEA vs SFA",
    ]
    if bool(time_trends.get("available", False)):
        tab_names.append("Time Trends")
    if bool(malmquist.get("available", False)):
        tab_names.append("Malmquist")
    tab_names.extend(
        [
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
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["Results"]:
        render_results_tab(results_df, results_df["DEA_Efficiency"].to_numpy(dtype=float), results_df["SFA_Efficiency"].to_numpy(dtype=float))
    with tabs["Targets & Peers"]:
        render_targets_tab(target_df, peer_weights, dmu_ids)
    with tabs["DEA vs SFA"]:
        render_comparison_tab(results_df)
    if "Time Trends" in tabs:
        with tabs["Time Trends"]:
            render_time_trends_tab(
                time_trends,
                str(config["time_col"]) if config.get("time_col") is not None else None,
                str(config.get("dea_time_mode", "")),
            )
    if "Malmquist" in tabs:
        with tabs["Malmquist"]:
            render_malmquist_tab(malmquist)
    with tabs["Distributions"]:
        render_distributions_tab(results_df)
    with tabs["Frontier"]:
        render_frontier_tab(df_clean, input_cols, output_cols, results_df)
    with tabs["Scale & RTS"]:
        render_scale_tab(results_df)
    with tabs["Benchmarks"]:
        render_benchmark_tab(support, cross_eff, dmu_ids)
    with tabs["Super & Robustness"]:
        render_super_and_robustness_tab(
            results_df,
            jackknife_df,
            dea_bootstrap,
            sfa_bootstrap,
            config["dea_orientation"],
            config["dea_returns"],
        )
    with tabs["SFA Details"]:
        render_sfa_tab_enhanced(sfa, sfa_param_df, str(sfa_output_col))
    with tabs["Diagnostics"]:
        render_diagnostics_tab(summary, errors, validation_warnings, results_df)
    with tabs["Export"]:
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
            time_trends,
            malmquist,
        )
