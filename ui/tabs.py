from __future__ import annotations

from typing import Dict, Optional, Sequence

try:
    import altair as alt
except ImportError as exc:
    raise ImportError(
        "This app uses Altair for interactive charts. Install `altair`; "
        "install `vl-convert-python` to enable PNG/SVG figure downloads."
    ) from exc

import numpy as np
import pandas as pd
import streamlit as st

from efficiency_tool.exports import make_excel_download
from efficiency_tool.ui.charts import render_altair_chart_with_downloads
from efficiency_tool.ui.widgets import display_download_button
from efficiency_tool.utils.stats import finite_corr, finite_spearman

try:
    alt.data_transformers.disable_max_rows()
except Exception:
    pass

def render_intro() -> None:
    st.info("Upload a CSV or Excel file to get started.")
    st.markdown(
        """
        ### What this version adds

        **DEA:** corrected output-oriented 0-1 efficiency scores, slacks, target input/output levels,
        peer weights, super-efficiency, CRS/VRS scale efficiency, returns-to-scale diagnostics,
        solver messages, and benchmark support.

        **SFA:** explicit single-output production or cost frontier, Cobb-Douglas or Translog specification,
        half-normal ML estimation, optional already-logged SFA variables for R/Stata matching,
        convergence diagnostics, gamma, AIC/BIC, and coefficient output.

        **Workflow:** DMU identifiers are preserved, DEA and SFA variables can be specified separately, optional metadata columns can be exported,
        expensive computations are cached, optional DEA bootstrap robustness is available,
        time-trend improvement summaries and DEA Malmquist productivity indices appear when panel data are selected,
        visualizations are interactive Altair charts, and downloads include results, targets, peers,
        bootstrap summaries, diagnostics, time summaries, Malmquist tables, and figure exports.
        """
    )

def render_validation(summary: Dict[str, object], warnings_list: Sequence[str]) -> None:
    removed_missing = int(summary["rows_removed_missing"])
    removed_nonpositive = int(summary["rows_removed_nonpositive"])
    if removed_missing or removed_nonpositive:
        st.warning(
            f"Removed {removed_missing + removed_nonpositive} rows before modeling: "
            f"{removed_missing} with missing/non-numeric model values and "
            f"{removed_nonpositive} with non-positive values in columns that require positive levels."
        )
    if int(summary.get("duplicate_ids", 0)) > 0:
        st.warning(
            f"Found {summary['duplicate_ids']} duplicate analysis-unit IDs after applying the selected "
            "DMU/time columns. Row numbers were appended to keep result keys unique."
        )
    for warning in warnings_list:
        st.warning(warning)

def render_results_tab(results_df: pd.DataFrame, dea_scores: np.ndarray, sfa_scores: np.ndarray) -> None:
    st.subheader("Efficiency Scores and Rankings")
    with st.expander("How to read these results", expanded=False):
        st.markdown(
            """
            **DEA_Efficiency is reported on a 0-1 scale**, where 1 means the unit is on the estimated DEA frontier.

            For output-oriented DEA, the LP estimates an output expansion factor phi. This app reports **1 / phi** in DEA_Efficiency and also exports **DEA_Radial_Factor** and **DEA_R_Benchmarking_Style_Score** for comparison with packages that report phi directly.

            **SFA_Efficiency** is usually 0-1. For cost frontiers, it can be reported as a Stata/FRONTIER-style cost ratio when that convention is selected.

            **Super-efficiency**, when available, can exceed 1 because it removes the evaluated unit from the reference set to rank frontier units.
            """
        )

    display_cols = [
        col
        for col in [
            "DMU_ID",
            "DEA_Efficiency",
            "DEA_Radial_Factor",
            "DEA_R_Benchmarking_Style_Score",
            "DEA_Rank",
            "DEA_R_Benchmarking_Style_Rank",
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
        plot_df = results_df.loc[mask, ["DMU_ID", "DEA_Efficiency", "SFA_Efficiency"]].copy()
        plot_df["DEA_Efficiency"] = pd.to_numeric(plot_df["DEA_Efficiency"], errors="coerce")
        plot_df["SFA_Efficiency"] = pd.to_numeric(plot_df["SFA_Efficiency"], errors="coerce")

        sfa_upper = float(np.nanmax(plot_df["SFA_Efficiency"])) if len(plot_df) else 1.0
        sfa_domain_upper = max(1.0, min(sfa_upper * 1.05, sfa_upper + 0.05))

        scatter = (
            alt.Chart(plot_df)
            .mark_circle(size=75, opacity=0.70)
            .encode(
                x=alt.X(
                    field="DEA_Efficiency",
                    type="quantitative",
                    title="DEA efficiency",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    field="SFA_Efficiency",
                    type="quantitative",
                    title="SFA efficiency",
                    scale=alt.Scale(domain=[0, sfa_domain_upper]),
                ),
                tooltip=[
                    alt.Tooltip(field="DMU_ID", type="nominal", title="DMU"),
                    alt.Tooltip(field="DEA_Efficiency", type="quantitative", title="DEA", format=".4f"),
                    alt.Tooltip(field="SFA_Efficiency", type="quantitative", title="SFA", format=".4f"),
                ],
            )
        )
        chart = scatter
        if sfa_domain_upper <= 1.05:
            identity_line = (
                alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
                .mark_line(strokeDash=[6, 4])
                .encode(
                    x=alt.X(field="x", type="quantitative", title="DEA efficiency"),
                    y=alt.Y(field="y", type="quantitative", title="SFA efficiency"),
                )
            )
            chart = scatter + identity_line
        chart = chart.properties(
            title="DEA vs SFA efficiency",
            height=420,
        ).interactive()
        render_altair_chart_with_downloads(chart, "dea_vs_sfa_efficiency", plot_df)

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
    for col in ["DEA_Efficiency", "DEA_R_Benchmarking_Style_Score", "SFA_Efficiency", "Scale_Efficiency"]:
        if col not in results_df:
            continue
        values = results_df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue

        plot_df = pd.DataFrame({col: values})
        n_bins = min(20, max(5, int(np.sqrt(len(values)))))
        mean_df = pd.DataFrame({col: [float(np.mean(values))], "Statistic": ["Mean"]})

        hist = (
            alt.Chart(plot_df)
            .mark_bar(opacity=0.75)
            .encode(
                x=alt.X(
                    field=col,
                    type="quantitative",
                    bin=alt.Bin(maxbins=n_bins),
                    title=col,
                ),
                y=alt.Y("count():Q", title="Frequency"),
                tooltip=[alt.Tooltip("count():Q", title="Frequency")],
            )
        )
        mean_rule = (
            alt.Chart(mean_df)
            .mark_rule(strokeDash=[6, 4], strokeWidth=2)
            .encode(
                x=alt.X(field=col, type="quantitative", title=col),
                tooltip=[
                    alt.Tooltip(field="Statistic", type="nominal", title="Statistic"),
                    alt.Tooltip(field=col, type="quantitative", title="Value", format=".4f"),
                ],
            )
        )
        chart = (hist + mean_rule).properties(
            title=f"{col} distribution",
            height=360,
        ).interactive()
        render_altair_chart_with_downloads(chart, f"{col}_distribution", plot_df)

def render_frontier_tab(df_clean: pd.DataFrame, input_cols: Sequence[str], output_cols: Sequence[str], results_df: pd.DataFrame) -> None:
    st.subheader("Frontier Visualization")
    if len(input_cols) == 1 and len(output_cols) == 1:
        input_col = input_cols[0]
        output_col = output_cols[0]
        eff = results_df["DEA_Efficiency"].to_numpy(dtype=float)
        frontier_mask = np.isfinite(eff) & (eff >= 0.999)

        plot_df = df_clean[[input_col, output_col]].copy()
        plot_df["DMU_ID"] = results_df["DMU_ID"].astype(str).to_numpy()
        plot_df["DEA_Efficiency"] = eff
        plot_df["DEA_Frontier_Status"] = np.where(frontier_mask, "DEA frontier", "Interior")
        plot_df = plot_df[np.isfinite(plot_df["DEA_Efficiency"].to_numpy(dtype=float))].copy()

        base_tooltips = [
            alt.Tooltip(field="DMU_ID", type="nominal", title="DMU"),
            alt.Tooltip(field=input_col, type="quantitative", title=input_col, format=".4f"),
            alt.Tooltip(field=output_col, type="quantitative", title=output_col, format=".4f"),
            alt.Tooltip(field="DEA_Efficiency", type="quantitative", title="DEA efficiency", format=".4f"),
            alt.Tooltip(field="DEA_Frontier_Status", type="nominal", title="Status"),
        ]
        points = (
            alt.Chart(plot_df)
            .mark_circle(size=80, opacity=0.80)
            .encode(
                x=alt.X(field=input_col, type="quantitative", title=input_col),
                y=alt.Y(field=output_col, type="quantitative", title=output_col),
                color=alt.Color(field="DEA_Efficiency", type="quantitative", title="DEA efficiency"),
                tooltip=base_tooltips,
            )
        )
        chart = points
        if frontier_mask.any():
            frontier_df = plot_df[plot_df["DEA_Frontier_Status"] == "DEA frontier"].copy()
            frontier_points = (
                alt.Chart(frontier_df)
                .mark_point(shape="star", size=220, filled=True)
                .encode(
                    x=alt.X(field=input_col, type="quantitative", title=input_col),
                    y=alt.Y(field=output_col, type="quantitative", title=output_col),
                    tooltip=base_tooltips,
                )
            )
            chart = points + frontier_points

        chart = chart.properties(
            title="Observed production set",
            height=420,
        ).interactive()
        render_altair_chart_with_downloads(chart, "observed_production_frontier", plot_df)
    else:
        st.info("Frontier visualization is shown when exactly one input and one output are selected.")

def render_scale_tab(results_df: pd.DataFrame) -> None:
    st.subheader("Scale Efficiency and Returns to Scale")
    scale_cols = ["DMU_ID", "DEA_CRS_Efficiency", "DEA_VRS_Efficiency", "Scale_Efficiency", "CRS_Lambda_Sum", "Returns_to_Scale"]
    st.dataframe(results_df[scale_cols].round(4), width="stretch")

    counts = results_df["Returns_to_Scale"].value_counts(dropna=False)
    if not counts.empty:
        plot_df = counts.rename_axis("Returns_to_Scale").reset_index(name="DMU_Count")
        plot_df["Returns_to_Scale"] = plot_df["Returns_to_Scale"].astype(str)
        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X(field="Returns_to_Scale", type="nominal", title="Returns-to-scale diagnostic", sort="-y"),
                y=alt.Y(field="DMU_Count", type="quantitative", title="Number of DMUs"),
                tooltip=[
                    alt.Tooltip(field="Returns_to_Scale", type="nominal", title="RTS"),
                    alt.Tooltip(field="DMU_Count", type="quantitative", title="DMUs", format=",d"),
                ],
            )
            .properties(title="Returns-to-scale profile", height=360)
            .interactive()
        )
        render_altair_chart_with_downloads(chart, "returns_to_scale_profile", plot_df)

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
            plot_df["DMU_ID"] = plot_df["DMU_ID"].astype(str)
            dmu_order = plot_df["DMU_ID"].tolist()
            plot_df = plot_df[
                [
                    "DMU_ID",
                    "DEA_Efficiency",
                    "DEA_Bootstrap_Mean",
                    "DEA_Bootstrap_CI_Lower",
                    "DEA_Bootstrap_CI_Upper",
                    "DEA_Bootstrap_Rank_Mean",
                ]
            ].copy()

            interval_rules = (
                alt.Chart(plot_df)
                .mark_rule()
                .encode(
                    x=alt.X(
                        field="DMU_ID",
                        type="nominal",
                        title="DMU",
                        sort=dmu_order,
                        axis=alt.Axis(labelAngle=-90),
                    ),
                    y=alt.Y(
                        field="DEA_Bootstrap_CI_Lower",
                        type="quantitative",
                        title="DEA bootstrap efficiency",
                        scale=alt.Scale(domain=[0, 1.05]),
                    ),
                    y2=alt.Y2(field="DEA_Bootstrap_CI_Upper"),
                    tooltip=[
                        alt.Tooltip(field="DMU_ID", type="nominal", title="DMU"),
                        alt.Tooltip(field="DEA_Efficiency", type="quantitative", title="Baseline DEA", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_Mean", type="quantitative", title="Bootstrap mean", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_CI_Lower", type="quantitative", title="CI lower", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_CI_Upper", type="quantitative", title="CI upper", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_Rank_Mean", type="quantitative", title="Mean rank", format=".2f"),
                    ],
                )
            )
            mean_points = (
                alt.Chart(plot_df)
                .mark_point(size=80, filled=True)
                .encode(
                    x=alt.X(
                        field="DMU_ID",
                        type="nominal",
                        title="DMU",
                        sort=dmu_order,
                        axis=alt.Axis(labelAngle=-90),
                    ),
                    y=alt.Y(
                        field="DEA_Bootstrap_Mean",
                        type="quantitative",
                        title="DEA bootstrap efficiency",
                        scale=alt.Scale(domain=[0, 1.05]),
                    ),
                    tooltip=[
                        alt.Tooltip(field="DMU_ID", type="nominal", title="DMU"),
                        alt.Tooltip(field="DEA_Efficiency", type="quantitative", title="Baseline DEA", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_Mean", type="quantitative", title="Bootstrap mean", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_CI_Lower", type="quantitative", title="CI lower", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_CI_Upper", type="quantitative", title="CI upper", format=".4f"),
                        alt.Tooltip(field="DEA_Bootstrap_Rank_Mean", type="quantitative", title="Mean rank", format=".2f"),
                    ],
                )
            )
            chart = (interval_rules + mean_points).properties(
                title="DEA bootstrap intervals for top baseline DEA performers",
                height=420,
            ).interactive()
            render_altair_chart_with_downloads(chart, "dea_bootstrap_intervals_top_performers", plot_df)
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

def render_time_trends_tab(time_trends: Dict[str, object], time_col: Optional[str], dea_time_mode: str = "") -> None:
    st.subheader("Improvement Over Time")
    if not bool(time_trends.get("available", False)):
        reason = str(time_trends.get("reason", "Select a time column with at least two periods to show time trends."))
        st.info(reason)
        return

    period_summary = time_trends.get("period_summary", pd.DataFrame())
    unit_change = time_trends.get("unit_change", pd.DataFrame())
    score_long = time_trends.get("score_long", pd.DataFrame())
    change_summary = time_trends.get("change_summary", pd.DataFrame())
    period_labels = list(time_trends.get("period_labels", []))
    unit_col = str(time_trends.get("unit_col", "unit"))
    selected_time_col = str(time_trends.get("time_col", time_col or "time"))

    st.markdown(
        "This section compares each base unit across the selected time periods. "
        "It summarizes first-to-last changes in the already-estimated DEA/SFA scores; "
        "it is not a Malmquist productivity index or a panel SFA model."
    )
    if dea_time_mode:
        st.caption(f"DEA time mode used for these score changes: {dea_time_mode}.")
    st.caption(f"Time column: {selected_time_col}. Unit column used for matching over time: {unit_col}.")

    if not isinstance(change_summary, pd.DataFrame) or change_summary.empty:
        st.info("There are multiple time periods, but no units have at least two usable score observations to compare.")
        if isinstance(period_summary, pd.DataFrame) and not period_summary.empty:
            st.write("**Average scores by period**")
            st.dataframe(period_summary.round(4), width="stretch")
        return

    score_options = change_summary[["Score", "Score_Label"]].drop_duplicates().sort_values("Score_Label")
    label_to_score = dict(zip(score_options["Score_Label"], score_options["Score"]))
    selected_label = st.selectbox("Score to inspect", options=score_options["Score_Label"].tolist(), key="time_trend_score_select")
    selected_score = label_to_score[selected_label]

    selected_summary = change_summary[change_summary["Score"] == selected_score]
    selected_change = unit_change[unit_change["Score"] == selected_score].copy()
    selected_periods = period_summary[period_summary["Score"] == selected_score].copy()
    selected_long = score_long[score_long["Score"] == selected_score].copy()

    if not selected_summary.empty:
        row = selected_summary.iloc[0]
        lower_is_better = not bool(row.get("Higher_Is_Better", True))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Periods", f"{len(period_labels):,}")
        col2.metric("Units compared", f"{int(row.get('Units_Compared', 0)):,}")
        col3.metric("Avg first score", f"{float(row.get('Average_First_Score', np.nan)):.4f}" if np.isfinite(row.get("Average_First_Score", np.nan)) else "NA")
        col4.metric("Avg last score", f"{float(row.get('Average_Last_Score', np.nan)):.4f}" if np.isfinite(row.get("Average_Last_Score", np.nan)) else "NA")
        col5.metric("Share improved", f"{100.0 * float(row.get('Share_Improved', np.nan)):.1f}%" if np.isfinite(row.get("Share_Improved", np.nan)) else "NA")
        if lower_is_better:
            st.caption("For this selected score, lower values are treated as improvement.")
        else:
            st.caption("For this selected score, higher values are treated as improvement.")

    if isinstance(selected_periods, pd.DataFrame) and not selected_periods.empty:
        selected_periods = selected_periods.sort_values("Time_Order", kind="mergesort")
        trend_chart = (
            alt.Chart(selected_periods)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    field="Time_Period",
                    type="nominal",
                    title="Time period",
                    sort=period_labels,
                ),
                y=alt.Y(field="Mean_Score", type="quantitative", title=f"Mean {selected_label}"),
                tooltip=[
                    alt.Tooltip(field="Time_Period", type="nominal", title="Period"),
                    alt.Tooltip(field="Units_With_Score", type="quantitative", title="Units", format=",d"),
                    alt.Tooltip(field="Mean_Score", type="quantitative", title="Mean", format=".4f"),
                    alt.Tooltip(field="Median_Score", type="quantitative", title="Median", format=".4f"),
                ],
            )
            .properties(title=f"Average {selected_label} over time", height=360)
            .interactive()
        )
        render_altair_chart_with_downloads(trend_chart, f"time_trend_mean_{selected_score}", selected_periods)

    if isinstance(selected_change, pd.DataFrame) and not selected_change.empty:
        top_n = min(15, len(selected_change))
        top_improvers = selected_change.sort_values("Improvement", ascending=False).head(top_n)
        top_decliners = selected_change.sort_values("Improvement", ascending=True).head(top_n)
        change_plot = pd.concat([top_improvers, top_decliners], ignore_index=True).drop_duplicates(
            subset=["Trend_Unit_ID", "Score"], keep="first"
        )
        change_plot = change_plot.sort_values("Improvement", ascending=True, kind="mergesort")
        change_plot["Change_Direction"] = np.where(change_plot["Improvement"] >= 0, "Improved", "Declined")

        change_chart = (
            alt.Chart(change_plot)
            .mark_bar()
            .encode(
                x=alt.X(field="Improvement", type="quantitative", title=f"First-to-last improvement in {selected_label}"),
                y=alt.Y(field="Trend_Unit_ID", type="nominal", title="Unit", sort=change_plot["Trend_Unit_ID"].tolist()),
                tooltip=[
                    alt.Tooltip(field="Trend_Unit_ID", type="nominal", title="Unit"),
                    alt.Tooltip(field="First_Period", type="nominal", title="First period"),
                    alt.Tooltip(field="First_Score", type="quantitative", title="First score", format=".4f"),
                    alt.Tooltip(field="Last_Period", type="nominal", title="Last period"),
                    alt.Tooltip(field="Last_Score", type="quantitative", title="Last score", format=".4f"),
                    alt.Tooltip(field="Improvement", type="quantitative", title="Improvement", format=".4f"),
                    alt.Tooltip(field="Improvement_Percent_of_First", type="quantitative", title="Improvement % of first", format=".2f"),
                ],
            )
            .properties(title=f"Largest improvements and declines: {selected_label}", height=max(320, 20 * len(change_plot)))
            .interactive()
        )
        render_altair_chart_with_downloads(change_chart, f"time_trend_unit_change_{selected_score}", change_plot)

        with st.expander("Score trajectories for top improvers", expanded=False):
            trajectory_units = top_improvers["Trend_Unit_ID"].head(min(12, len(top_improvers))).tolist()
            trajectory_df = selected_long[selected_long["Trend_Unit_ID"].isin(trajectory_units)].copy()
            if not trajectory_df.empty:
                trajectory_df = trajectory_df.sort_values(["Trend_Unit_ID", "Time_Order"], kind="mergesort")
                trajectory_chart = (
                    alt.Chart(trajectory_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(field="Time_Period", type="nominal", title="Time period", sort=period_labels),
                        y=alt.Y(field="Score_Value", type="quantitative", title=selected_label),
                        color=alt.Color(field="Trend_Unit_ID", type="nominal", title="Unit"),
                        tooltip=[
                            alt.Tooltip(field="Trend_Unit_ID", type="nominal", title="Unit"),
                            alt.Tooltip(field="Time_Period", type="nominal", title="Period"),
                            alt.Tooltip(field="Score_Value", type="quantitative", title="Score", format=".4f"),
                        ],
                    )
                    .properties(title=f"Top improver trajectories: {selected_label}", height=420)
                    .interactive()
                )
                render_altair_chart_with_downloads(trajectory_chart, f"time_trend_top_trajectories_{selected_score}", trajectory_df)
            else:
                st.info("No trajectory data are available for the selected score.")

        st.write("**Unit-level first-to-last changes**")
        display_cols = [
            "Trend_Unit_ID",
            "First_Period",
            "First_Score",
            "Last_Period",
            "Last_Score",
            "Raw_Change",
            "Improvement",
            "Improvement_Percent_of_First",
            "Improved",
            "Declined",
            "Periods_Observed",
        ]
        display_cols = [col for col in display_cols if col in selected_change.columns]
        st.dataframe(selected_change[display_cols].round(4), width="stretch")

    with st.expander("All time-trend summary tables", expanded=False):
        st.write("**Period summary**")
        if isinstance(period_summary, pd.DataFrame) and not period_summary.empty:
            st.dataframe(period_summary.round(4), width="stretch")
        st.write("**Change summary by score**")
        if isinstance(change_summary, pd.DataFrame) and not change_summary.empty:
            st.dataframe(change_summary.round(4), width="stretch")

def render_malmquist_tab(malmquist: Dict[str, object]) -> None:
    st.subheader("Malmquist Productivity Index")
    if not bool(malmquist.get("available", False)):
        reason = str(malmquist.get("reason", "Select a DMU identifier and time column with repeated units to calculate Malmquist indices."))
        st.info(reason)
        return

    summary = malmquist.get("summary", pd.DataFrame())
    pairwise = malmquist.get("pairwise", pd.DataFrame())
    chain = malmquist.get("chain", pd.DataFrame())
    period_pairs = malmquist.get("period_pairs", pd.DataFrame())

    if not isinstance(summary, pd.DataFrame) or summary.empty or not isinstance(pairwise, pd.DataFrame) or pairwise.empty:
        st.info("No finite Malmquist results are available.")
        return

    st.markdown(
        "Malmquist decomposes productivity change into **efficiency change (EC)** and "
        "**technical change (TC)**. The app reports **MPI = EC × TC**; values above 1 indicate productivity improvement."
    )
    st.caption(
        f"Specification: {str(malmquist.get('orientation', '')).upper()} orientation, "
        f"{str(malmquist.get('returns', '')).upper()} returns to scale. "
        f"Common-reference mode: {bool(malmquist.get('same_reference', False))}."
    )

    finite_pairwise = pairwise[np.isfinite(pd.to_numeric(pairwise["Malmquist_MPI"], errors="coerce"))].copy()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean MPI", f"{finite_pairwise['Malmquist_MPI'].mean():.4f}" if not finite_pairwise.empty else "NA")
    col2.metric("Share improved", f"{100.0 * finite_pairwise['Productivity_Improved'].mean():.1f}%" if not finite_pairwise.empty else "NA")
    col3.metric("Mean EC", f"{finite_pairwise['Efficiency_Change_EC'].mean():.4f}" if not finite_pairwise.empty else "NA")
    col4.metric("Mean TC", f"{finite_pairwise['Technical_Change_TC'].mean():.4f}" if not finite_pairwise.empty else "NA")

    if isinstance(period_pairs, pd.DataFrame) and not period_pairs.empty:
        st.write("**Adjacent period feasibility and matching**")
        st.dataframe(period_pairs.round(4), width="stretch")

    st.write("**Period-pair summary**")
    st.dataframe(summary.round(4), width="stretch")

    summary_plot = summary.copy()
    summary_plot["Period_Pair"] = summary_plot["Period_0"].astype(str) + " → " + summary_plot["Period_1"].astype(str)
    metric_cols = ["Mean_MPI", "Mean_Efficiency_Change", "Mean_Technical_Change"]
    available_metrics = [col for col in metric_cols if col in summary_plot.columns]
    if available_metrics:
        long_summary = summary_plot.melt(
            id_vars=["Period_Pair", "Period_1_Order"],
            value_vars=available_metrics,
            var_name="Component",
            value_name="Index_Value",
        )
        long_summary["Component"] = long_summary["Component"].replace(
            {
                "Mean_MPI": "MPI",
                "Mean_Efficiency_Change": "Efficiency change",
                "Mean_Technical_Change": "Technical change",
            }
        )
        chart = (
            alt.Chart(long_summary)
            .mark_line(point=True)
            .encode(
                x=alt.X(field="Period_Pair", type="nominal", title="Adjacent period pair", sort=summary_plot["Period_Pair"].tolist()),
                y=alt.Y(field="Index_Value", type="quantitative", title="Mean index value"),
                color=alt.Color(field="Component", type="nominal", title="Component"),
                tooltip=[
                    alt.Tooltip(field="Period_Pair", type="nominal", title="Period pair"),
                    alt.Tooltip(field="Component", type="nominal", title="Component"),
                    alt.Tooltip(field="Index_Value", type="quantitative", title="Index", format=".4f"),
                ],
            )
            .properties(title="Malmquist decomposition by adjacent period", height=380)
            .interactive()
        )
        render_altair_chart_with_downloads(chart, "malmquist_period_decomposition", long_summary)

    pair_options = summary_plot["Period_Pair"].tolist()
    selected_pair = st.selectbox("Inspect adjacent period pair", options=pair_options, key="malmquist_pair_select")
    selected_row = summary_plot[summary_plot["Period_Pair"] == selected_pair].iloc[0]
    selected_pairwise = pairwise[
        (pairwise["Period_0"].astype(str) == str(selected_row["Period_0"]))
        & (pairwise["Period_1"].astype(str) == str(selected_row["Period_1"]))
    ].copy()
    selected_pairwise = selected_pairwise[np.isfinite(pd.to_numeric(selected_pairwise["Malmquist_MPI"], errors="coerce"))].copy()

    if not selected_pairwise.empty:
        selected_pairwise["MPI_Distance_From_1"] = selected_pairwise["Malmquist_MPI"] - 1.0
        top_n = min(15, max(1, len(selected_pairwise) // 2))
        improvers = selected_pairwise.nlargest(top_n, "Malmquist_MPI")
        decliners = selected_pairwise.nsmallest(top_n, "Malmquist_MPI")
        change_plot = pd.concat([improvers, decliners], ignore_index=True).drop_duplicates("Trend_Unit_ID")
        change_plot = change_plot.sort_values("Malmquist_MPI", ascending=True, kind="mergesort")
        unit_order = change_plot["Trend_Unit_ID"].astype(str).tolist()
        bars = (
            alt.Chart(change_plot)
            .mark_bar()
            .encode(
                y=alt.Y(field="Trend_Unit_ID", type="nominal", title="Unit", sort=unit_order),
                x=alt.X(field="Malmquist_MPI", type="quantitative", title="Malmquist MPI"),
                tooltip=[
                    alt.Tooltip(field="Trend_Unit_ID", type="nominal", title="Unit"),
                    alt.Tooltip(field="Malmquist_MPI", type="quantitative", title="MPI", format=".4f"),
                    alt.Tooltip(field="Efficiency_Change_EC", type="quantitative", title="EC", format=".4f"),
                    alt.Tooltip(field="Technical_Change_TC", type="quantitative", title="TC", format=".4f"),
                    alt.Tooltip(field="Productivity_Change_Percent", type="quantitative", title="MPI change %", format=".2f"),
                ],
            )
        )
        one_rule = (
            alt.Chart(pd.DataFrame({"x": [1.0]}))
            .mark_rule(strokeDash=[6, 4])
            .encode(x=alt.X(field="x", type="quantitative"))
        )
        chart = (bars + one_rule).properties(title=f"Unit MPI: {selected_pair}", height=max(320, 22 * len(change_plot))).interactive()
        render_altair_chart_with_downloads(chart, "malmquist_unit_mpi_selected_pair", change_plot)

        display_cols = [
            "Trend_Unit_ID",
            "Malmquist_MPI",
            "Efficiency_Change_EC",
            "Technical_Change_TC",
            "Productivity_Change_Percent",
            "Productivity_Improved",
            "Productivity_Declined",
            "e00_Period0_On_Period0_Frontier",
            "e01_Period0_On_Period1_Frontier",
            "e10_Period1_On_Period0_Frontier",
            "e11_Period1_On_Period1_Frontier",
            "All_Distance_LPs_Succeeded",
        ]
        st.write("**Unit-level Malmquist results for selected pair**")
        st.dataframe(selected_pairwise[display_cols].round(4), width="stretch")
    else:
        st.info("No finite unit-level MPI values are available for the selected period pair.")

    if isinstance(chain, pd.DataFrame) and not chain.empty and chain["MPI_Chain_Index"].notna().any():
        st.write("**Malmquist chain index**")
        latest = chain.sort_values("Time_Order", kind="mergesort").groupby("Trend_Unit_ID", dropna=False).tail(1)
        top_units = latest.nlargest(min(20, len(latest)), "MPI_Chain_Index")["Trend_Unit_ID"].astype(str).tolist()
        chain_plot = chain[chain["Trend_Unit_ID"].astype(str).isin(top_units)].copy()
        chart = (
            alt.Chart(chain_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X(field="Time_Period", type="nominal", title="Time period", sort=list(malmquist.get("period_labels", []))),
                y=alt.Y(field="MPI_Chain_Index", type="quantitative", title="MPI chain index"),
                color=alt.Color(field="Trend_Unit_ID", type="nominal", title="Unit"),
                tooltip=[
                    alt.Tooltip(field="Trend_Unit_ID", type="nominal", title="Unit"),
                    alt.Tooltip(field="Time_Period", type="nominal", title="Period"),
                    alt.Tooltip(field="MPI_Chain_Index", type="quantitative", title="MPI chain", format=".4f"),
                    alt.Tooltip(field="Adjacent_MPI", type="quantitative", title="Adjacent MPI", format=".4f"),
                ],
            )
            .properties(title="Top Malmquist chain-index trajectories", height=420)
            .interactive()
        )
        render_altair_chart_with_downloads(chart, "malmquist_chain_index_trajectories", chain_plot)
        st.dataframe(chain.round(4), width="stretch")

    with st.expander("All Malmquist tables", expanded=False):
        st.write("**Unit-level adjacent-period MPI**")
        st.dataframe(pairwise.round(4), width="stretch")
        if isinstance(chain, pd.DataFrame) and not chain.empty:
            st.write("**Chain index**")
            st.dataframe(chain.round(4), width="stretch")

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
    time_trends: Optional[Dict[str, object]] = None,
    malmquist: Optional[Dict[str, object]] = None,
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

    if time_trends is not None and bool(time_trends.get("available", False)):
        time_unit_change = time_trends.get("unit_change")
        if isinstance(time_unit_change, pd.DataFrame) and not time_unit_change.empty:
            display_download_button(
                "Download time improvement CSV",
                time_unit_change.to_csv(index=False).encode("utf-8"),
                "time_improvement_by_unit.csv",
                "text/csv",
            )
        time_period_summary = time_trends.get("period_summary")
        if isinstance(time_period_summary, pd.DataFrame) and not time_period_summary.empty:
            display_download_button(
                "Download time period summary CSV",
                time_period_summary.to_csv(index=False).encode("utf-8"),
                "time_period_summary.csv",
                "text/csv",
            )

    if malmquist is not None and bool(malmquist.get("available", False)):
        malm_pairwise = malmquist.get("pairwise")
        if isinstance(malm_pairwise, pd.DataFrame) and not malm_pairwise.empty:
            display_download_button(
                "Download Malmquist unit MPI CSV",
                malm_pairwise.to_csv(index=False).encode("utf-8"),
                "malmquist_unit_mpi.csv",
                "text/csv",
            )
        malm_summary = malmquist.get("summary")
        if isinstance(malm_summary, pd.DataFrame) and not malm_summary.empty:
            display_download_button(
                "Download Malmquist period summary CSV",
                malm_summary.to_csv(index=False).encode("utf-8"),
                "malmquist_period_summary.csv",
                "text/csv",
            )

    tables: Dict[str, pd.DataFrame] = {
        "Results": results_df,
        "DEA_Targets": target_df,
        "Peer_Weights": peer_weights,
        "Benchmark_Support": support,
        "SFA_Parameters": sfa_param_df,
    }
    if time_trends is not None and bool(time_trends.get("available", False)):
        time_period_summary = time_trends.get("period_summary")
        time_unit_change = time_trends.get("unit_change")
        time_score_long = time_trends.get("score_long")
        time_change_summary = time_trends.get("change_summary")
        if isinstance(time_period_summary, pd.DataFrame) and not time_period_summary.empty:
            tables["Time_Period_Summary"] = time_period_summary
        if isinstance(time_change_summary, pd.DataFrame) and not time_change_summary.empty:
            tables["Time_Change_Summary"] = time_change_summary
        if isinstance(time_unit_change, pd.DataFrame) and not time_unit_change.empty:
            tables["Time_Unit_Change"] = time_unit_change
        if isinstance(time_score_long, pd.DataFrame) and not time_score_long.empty:
            tables["Time_Score_Long"] = time_score_long
    if malmquist is not None and bool(malmquist.get("available", False)):
        malm_summary = malmquist.get("summary")
        malm_pairwise = malmquist.get("pairwise")
        malm_chain = malmquist.get("chain")
        malm_pairs = malmquist.get("period_pairs")
        if isinstance(malm_summary, pd.DataFrame) and not malm_summary.empty:
            tables["Malmquist_Summary"] = malm_summary
        if isinstance(malm_pairwise, pd.DataFrame) and not malm_pairwise.empty:
            tables["Malmquist_Unit_MPI"] = malm_pairwise
        if isinstance(malm_chain, pd.DataFrame) and not malm_chain.empty:
            tables["Malmquist_Chain"] = malm_chain
        if isinstance(malm_pairs, pd.DataFrame) and not malm_pairs.empty:
            tables["Malmquist_Pairs"] = malm_pairs
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

