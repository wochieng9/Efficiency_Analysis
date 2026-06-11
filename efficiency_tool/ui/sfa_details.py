from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from efficiency_tool.sfa.inference import sfa_equation_latex, sfa_returns_to_scale_test

def render_sfa_tab_enhanced(sfa: dict, sfa_param_df: pd.DataFrame, output_col: str) -> None:
    """
    Enhanced SFA regression output with interpretation, significance, and diagnostics.
    """
    st.subheader("SFA Frontier Estimation Results")

    # ========== Model Status & Fit ==========
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        status_text = "✓ Converged" if sfa.get("converged", False) else "⚠ Did not converge"
        st.metric("Convergence", status_text)
    with col2:
        st.metric(
            "Log Likelihood",
            f"{sfa.get('log_likelihood', np.nan):.2f}" if np.isfinite(sfa.get("log_likelihood", np.nan)) else "NA",
        )
    with col3:
        st.metric(
            "AIC",
            f"{sfa.get('aic', np.nan):.2f}" if np.isfinite(sfa.get("aic", np.nan)) else "NA",
        )
    with col4:
        st.metric(
            "BIC",
            f"{sfa.get('bic', np.nan):.2f}" if np.isfinite(sfa.get("bic", np.nan)) else "NA",
        )
    with col5:
        st.metric(
            "Gamma (σ²ᵤ/σ²)",
            f"{sfa.get('gamma', np.nan):.4f}" if np.isfinite(sfa.get("gamma", np.nan)) else "NA",
        )

    if not sfa.get("converged", False):
        st.warning(f"**Convergence note:** {sfa.get('message', 'Model did not converge.')}")

    with st.expander("SFA specification used", expanded=False):
        st.write({
            "Input/regressor columns": sfa.get("input_cols", []),
            "Dependent/output column": sfa.get("output_col", output_col),
            "Variables already logged": bool(sfa.get("data_are_logged", False)),
            "Frontier type": sfa.get("frontier_type", ""),
            "Cost efficiency convention": sfa.get("cost_efficiency_convention", ""),
        })
        if bool(sfa.get("data_are_logged", False)):
            st.caption("The selected SFA columns were used as logged variables directly. No additional natural log was applied.")
        else:
            st.caption("The selected SFA columns were treated as positive level variables and natural logs were applied internally.")

    # ========== Equation ==========
    st.subheader("Estimated Equation")
    frontier_type = str(sfa.get("frontier_type", "Production"))
    model_type = str(sfa.get("model_type", "Cobb-Douglas"))

    equation = sfa_equation_latex(sfa, output_col, frontier_type)
    st.markdown(f"**{model_type} {frontier_type} Frontier:**")
    st.code(equation, language="text")

    if frontier_type.lower() == "production":
        st.caption("where v is a symmetric noise term and u ≥ 0 is inefficiency (higher u → lower efficiency).")
    else:
        st.caption("where v is a symmetric noise term and u ≥ 0 is inefficiency (higher u → higher cost). Cost efficiency may be shown as a 0-1 reciprocal or as a cost ratio depending on the selected convention.")

    # ========== Coefficient Table ==========
    st.subheader("Frontier Coefficients")
    coeff_table = sfa_param_df.copy()

    if not coeff_table.empty:
        # Display with formatting
        display_table = coeff_table.copy()
        for col in ["Coefficient", "Std_Error", "t_stat", "p_value", "CI_Lower_95%", "CI_Upper_95%", "Elasticity"]:
            if col in display_table.columns:
                display_table[col] = display_table[col].apply(lambda x: f"{x:.6f}" if np.isfinite(x) else "NA")

        st.dataframe(display_table, width="content", height=400)

        st.caption(
            "**t_stat** is coefficient/Std_Error (approx. standard normal under H₀). "
            "**p_value** is two-tailed significance. "
            "**Elasticity** shows output response to 1% input change (for Cobb-Douglas first-order terms). "
            "**Standard errors** are approximate from optimizer Hessian."
        )
    else:
        st.info("No coefficient estimates available (model did not estimate successfully).")

    # ========== Variance Decomposition ==========
    with st.expander("**Variance Decomposition**", expanded=False):
        sigma_u = float(sfa.get("sigma_u", np.nan))
        sigma_v = float(sfa.get("sigma_v", np.nan))
        gamma = float(sfa.get("gamma", np.nan))

        if np.isfinite(sigma_u) and np.isfinite(sigma_v):
            sigma_total = np.sqrt(sigma_u**2 + sigma_v**2)
            variance_table = pd.DataFrame({
                "Component": ["σᵤ (inefficiency std.)", "σᵥ (noise std.)", "σ (total std.)", "γ = σ²ᵤ/σ² (signal fraction)"],
                "Value": [sigma_u, sigma_v, sigma_total, gamma],
            })
            st.dataframe(variance_table, width="content", hide_index=True)

            if np.isfinite(gamma):
                st.caption(
                    f"Gamma = {gamma:.4f} means {100*gamma:.1f}% of composite error is due to inefficiency; "
                    f"{100*(1-gamma):.1f}% is random noise."
                )
        else:
            st.info("Variance components not available.")

    # ========== Returns to Scale Test ==========
    with st.expander("**Returns to Scale (CRS) Test**", expanded=False):
        rts_test = sfa_returns_to_scale_test(sfa)

        if rts_test["test"] == "unavailable":
            st.info(rts_test["conclusion"])
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Σ Elasticities", f"{rts_test['sum_elasticity']:.4f}" if np.isfinite(rts_test["sum_elasticity"]) else "NA")
            with col2:
                st.metric("Std Error (Σ)", f"{rts_test['se_sum']:.4f}" if np.isfinite(rts_test["se_sum"]) else "NA")
            with col3:
                st.metric("t-statistic", f"{rts_test['t_stat']:.4f}" if np.isfinite(rts_test["t_stat"]) else "NA")
            with col4:
                st.metric("p-value (H₀: Σ=1)", f"{rts_test['p_value']:.4f}" if np.isfinite(rts_test["p_value"]) else "NA")

            st.markdown(f"**Conclusion:** {rts_test['conclusion']}")
            st.caption(
                "H₀: Constant returns to scale (sum of input elasticities = 1). "
                "Reject H₀ if p < 0.05."
            )

    # ========== Interpretation Guide ==========
    with st.expander("**How to Read These Results**", expanded=False):
        st.markdown("""
        **Coefficients:**
        - For Cobb-Douglas log-linear models, coefficients are direct output elasticities
        - A coefficient of 0.5 on ln(labor) means: 1% increase in labor → 0.5% increase in output

        **Standard Errors & t-statistics:**
        - Larger t-stat (|t| > 2) or smaller p-value (< 0.05) suggest the input is statistically significant
        - Standard errors here are *approximate* (from optimizer Hessian), not bootstrap-validated

        **Gamma (γ):**
        - Fraction of total variance due to inefficiency (not noise)
        - γ ≈ 0 means mostly random noise (little systematic inefficiency)
        - γ ≈ 1 means mostly true inefficiency

        **Returns to Scale (RTS) Test:**
        - If Σ elasticities ≈ 1: constant returns to scale (doubling inputs doubles output)
        - If Σ elasticities > 1: increasing returns (economies of scale)
        - If Σ elasticities < 1: decreasing returns (diseconomies of scale)
        """)

