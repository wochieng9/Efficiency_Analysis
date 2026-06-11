from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from efficiency_tool.dea.bootstrap import calculate_dea_bootstrap
from efficiency_tool.dea.malmquist import calculate_malmquist_indices
from efficiency_tool.dea.core import (
    calculate_cross_efficiency_crs,
    calculate_dea_with_slacks,
    calculate_dea_with_slacks_by_group,
    calculate_jackknife_influence,
    calculate_scale_diagnostics,
    calculate_scale_diagnostics_by_group,
    calculate_super_efficiency,
    calculate_super_efficiency_by_group,
)
from efficiency_tool.sfa.bootstrap import calculate_sfa_bootstrap
from efficiency_tool.sfa.core import calculate_sfa_production
from efficiency_tool.utils.data import load_uploaded_data as _load_uploaded_data


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    return _load_uploaded_data(file_name, file_bytes)


@st.cache_data(show_spinner=False)
def calculate_dea_with_slacks_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str = "output",
    returns: str = "crs",
    reference_group_col: Optional[str] = None,
) -> Dict[str, object]:
    if reference_group_col:
        return calculate_dea_with_slacks_by_group(data, reference_group_col, list(input_cols), list(output_cols), orientation, returns)
    return calculate_dea_with_slacks(data, list(input_cols), list(output_cols), orientation, returns)


@st.cache_data(show_spinner=False)
def calculate_super_efficiency_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str = "output",
    returns: str = "crs",
    reference_group_col: Optional[str] = None,
) -> Dict[str, object]:
    if reference_group_col:
        return calculate_super_efficiency_by_group(data, reference_group_col, list(input_cols), list(output_cols), orientation, returns)
    return calculate_super_efficiency(data, list(input_cols), list(output_cols), orientation, returns)


@st.cache_data(show_spinner=False)
def calculate_scale_diagnostics_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str,
    reference_group_col: Optional[str] = None,
) -> Dict[str, object]:
    if reference_group_col:
        return calculate_scale_diagnostics_by_group(data, reference_group_col, list(input_cols), list(output_cols), orientation)
    return calculate_scale_diagnostics(data, list(input_cols), list(output_cols), orientation)


@st.cache_data(show_spinner=False)
def calculate_malmquist_indices_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    unit_col: Optional[str],
    time_col: Optional[str],
    orientation: str,
    returns: str,
    same_reference: bool = False,
) -> Dict[str, object]:
    return calculate_malmquist_indices(
        data,
        list(input_cols),
        list(output_cols),
        unit_col,
        time_col,
        orientation=orientation,
        returns=returns,
        same_reference=bool(same_reference),
    )


@st.cache_data(show_spinner=False)
def calculate_cross_efficiency_crs_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    weight_floor: float = 0.0,
) -> Dict[str, object]:
    return calculate_cross_efficiency_crs(data, list(input_cols), list(output_cols), weight_floor)


@st.cache_data(show_spinner=False)
def calculate_jackknife_influence_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str,
    returns: str,
) -> pd.DataFrame:
    return calculate_jackknife_influence(data, list(input_cols), list(output_cols), orientation, returns)


@st.cache_data(show_spinner=False)
def calculate_dea_bootstrap_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_cols: Tuple[str, ...],
    orientation: str,
    returns: str,
    n_boot: int,
    ci_level: float,
    seed: int,
    include_self: bool,
) -> Dict[str, object]:
    return calculate_dea_bootstrap(
        data,
        list(input_cols),
        list(output_cols),
        orientation,
        returns,
        n_boot=int(n_boot),
        ci_level=float(ci_level),
        seed=int(seed),
        include_self=bool(include_self),
    )


@st.cache_data(show_spinner=False)
def calculate_sfa_production_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_col: str,
    model_type: str = "Cobb-Douglas",
    frontier_type: str = "Production",
    data_are_logged: bool = False,
    cost_efficiency_convention: str = "0-1 reciprocal E[exp(-u)|eps]",
) -> Dict[str, object]:
    return calculate_sfa_production(
        data,
        list(input_cols),
        output_col,
        model_type,
        frontier_type,
        data_are_logged=data_are_logged,
        cost_efficiency_convention=cost_efficiency_convention,
    )


@st.cache_data(show_spinner=False)
def calculate_sfa_bootstrap_cached(
    data: pd.DataFrame,
    input_cols: Tuple[str, ...],
    output_col: str,
    model_type: str,
    frontier_type: str,
    n_boot: int,
    ci_level: float,
    seed: int,
    data_are_logged: bool = False,
    cost_efficiency_convention: str = "0-1 reciprocal E[exp(-u)|eps]",
) -> Dict[str, object]:
    return calculate_sfa_bootstrap(
        data,
        list(input_cols),
        output_col,
        model_type,
        frontier_type,
        int(n_boot),
        float(ci_level),
        int(seed),
        data_are_logged=bool(data_are_logged),
        cost_efficiency_convention=cost_efficiency_convention,
    )
