from __future__ import annotations

import io
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def unique_keep_order(items: Iterable[Optional[str]]) -> List[str]:
    """Return unique non-empty column names while preserving order."""
    seen = set()
    out: List[str] = []
    for item in items:
        if item is None or item == "":
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def load_uploaded_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file from bytes."""
    suffix = file_name.lower().split(".")[-1]
    buffer = io.BytesIO(file_bytes)
    if suffix == "csv":
        return pd.read_csv(buffer)
    return pd.read_excel(buffer)


def likely_numeric_columns(df: pd.DataFrame, threshold: float = 0.80) -> List[str]:
    """Find columns that are probably usable as numeric model variables."""
    candidates: List[str] = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        share_numeric = converted.notna().mean() if len(converted) else 0
        if share_numeric >= threshold:
            candidates.append(col)
    return candidates


def format_identifier_component(value: object) -> str:
    """Format one ID or time value for stable DMU labels."""
    if pd.isna(value):
        return "<missing>"

    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return "<missing>"
        if timestamp == timestamp.normalize():
            return timestamp.strftime("%Y-%m-%d")
        return timestamp.isoformat()

    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))

    return str(value)


def build_analysis_dmu_ids(clean: pd.DataFrame, id_col: Optional[str], time_col: Optional[str]) -> Tuple[pd.Series, str]:
    """Build the unique analysis-unit label used by DEA/SFA outputs."""
    if id_col is not None and id_col in clean.columns:
        base_ids = clean[id_col].map(format_identifier_component)
        if time_col is not None and time_col in clean.columns and time_col != id_col:
            periods = clean[time_col].map(format_identifier_component)
            return base_ids + " | " + periods, f"{id_col} + {time_col}"
        return base_ids, str(id_col)

    return clean["Original_Row"].map(lambda x: f"DMU_{int(x)}"), "Original_Row"


def clean_and_prepare_data(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    id_col: Optional[str],
    time_col: Optional[str],
    group_col: Optional[str],
    env_cols: Sequence[str],
    selected_groups: Optional[Sequence[object]] = None,
    sfa_input_cols: Optional[Sequence[str]] = None,
    sfa_output_col: Optional[str] = None,
    sfa_data_are_logged: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Preserve metadata, coerce model columns to numeric, and drop invalid model rows.

    DEA columns must be strictly positive. SFA columns must be strictly positive when
    the app is asked to log them internally; when the user selects already logged SFA
    variables, they only need to be finite numeric values.
    """
    dea_cols = unique_keep_order(list(input_cols) + list(output_cols))
    sfa_cols = unique_keep_order(list(sfa_input_cols or []) + ([sfa_output_col] if sfa_output_col else []))
    model_cols = unique_keep_order(dea_cols + sfa_cols)
    metadata_cols = unique_keep_order([id_col, time_col, group_col] + list(env_cols))
    keep_cols = unique_keep_order(metadata_cols + model_cols)

    working = df.loc[:, keep_cols].copy()
    working["Original_Row"] = df.index + 1

    if group_col is not None and selected_groups:
        working = working[working[group_col].isin(selected_groups)].copy()

    for col in model_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    missing_mask = working[model_cols].isna().any(axis=1) if model_cols else pd.Series(False, index=working.index)

    positive_required_cols = list(dea_cols)
    if not sfa_data_are_logged:
        positive_required_cols = unique_keep_order(positive_required_cols + sfa_cols)

    if positive_required_cols:
        nonpositive_mask = (working[positive_required_cols] <= 0).any(axis=1)
    else:
        nonpositive_mask = pd.Series(False, index=working.index)

    invalid_mask = missing_mask | nonpositive_mask

    clean = working.loc[~invalid_mask].copy()
    dmu_ids, dmu_id_basis = build_analysis_dmu_ids(clean, id_col, time_col)
    clean["DMU_ID"] = dmu_ids.astype(str)

    duplicate_mask = clean["DMU_ID"].duplicated(keep=False)
    duplicate_ids_before_fix = int(clean["DMU_ID"].duplicated().sum())
    if duplicate_ids_before_fix > 0:
        clean.loc[duplicate_mask, "DMU_ID"] = (
            clean.loc[duplicate_mask, "DMU_ID"].astype(str)
            + " (row "
            + clean.loc[duplicate_mask, "Original_Row"].astype(int).astype(str)
            + ")"
        )

    summary: Dict[str, object] = {
        "rows_original": int(len(df)),
        "rows_after_group_filter": int(len(working)),
        "rows_removed_missing": int(missing_mask.sum()),
        "rows_removed_nonpositive": int((nonpositive_mask & ~missing_mask).sum()),
        "rows_clean": int(len(clean)),
        "dmu_id_basis": dmu_id_basis,
        "duplicate_ids": duplicate_ids_before_fix,
        "dea_positive_required_columns": positive_required_cols,
        "sfa_variables_treated_as_logged": bool(sfa_data_are_logged),
    }
    return clean.reset_index(drop=True), summary


def validate_analysis_setup(
    df_clean: pd.DataFrame,
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    id_col: Optional[str],
    sfa_model: str,
    sfa_input_cols: Optional[Sequence[str]] = None,
    sfa_output_col: Optional[str] = None,
    sfa_data_are_logged: bool = False,
    time_col: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Return blocking errors and non-blocking warnings for the selected setup."""
    errors: List[str] = []
    warnings: List[str] = []

    dea_input_cols = list(input_cols)
    dea_output_cols = list(output_cols)
    sfa_inputs = list(sfa_input_cols if sfa_input_cols is not None else input_cols)
    sfa_output = sfa_output_col if sfa_output_col is not None else (output_cols[0] if output_cols else None)

    overlap = sorted(set(dea_input_cols).intersection(dea_output_cols))
    if overlap:
        errors.append("DEA input and output columns must be disjoint: " + ", ".join(overlap))

    if not dea_input_cols:
        errors.append("Select at least one DEA input column.")
    if not dea_output_cols:
        errors.append("Select at least one DEA output column.")

    if not sfa_inputs:
        errors.append("Select at least one SFA input/regressor column.")
    if not sfa_output:
        errors.append("Select one SFA dependent/output column.")
    if sfa_output is not None and sfa_output in sfa_inputs:
        errors.append("The SFA dependent/output column cannot also be an SFA input/regressor.")

    if len(df_clean) == 0:
        errors.append("No usable rows remain after cleaning missing and invalid model values.")

    if id_col is not None and "DMU_ID" in df_clean and df_clean["DMU_ID"].duplicated().any():
        warnings.append("Duplicate analysis-unit identifiers were found. Rankings and peer tables may be harder to interpret.")

    model_cols = unique_keep_order(dea_input_cols + dea_output_cols + sfa_inputs + ([sfa_output] if sfa_output else []))
    for col in model_cols:
        if col in df_clean and df_clean[col].nunique(dropna=True) <= 1:
            warnings.append(f"Column '{col}' is constant after cleaning; it may weaken DEA/SFA identification.")

    n = len(df_clean)
    m = len(dea_input_cols)
    s = len(dea_output_cols)
    if n > 0 and m > 0 and s > 0:
        rule_of_thumb = max(m * s, 3 * (m + s))
        if n < rule_of_thumb:
            warnings.append(
                f"DEA has {n} DMUs for {m} inputs and {s} outputs. "
                f"A common rule of thumb is at least max(m*s, 3*(m+s)) = {rule_of_thumb} DMUs."
            )

        dea_cols = unique_keep_order(dea_input_cols + dea_output_cols)
        positive_dea_cols = [col for col in dea_cols if col in df_clean and (df_clean[col] > 0).all()]
        if positive_dea_cols:
            log_values = np.log(df_clean[positive_dea_cols].astype(float))
            zscores = (log_values - log_values.mean()) / log_values.std(ddof=0).replace(0, np.nan)
            outlier_rows = zscores.abs().gt(3).any(axis=1).sum()
            if outlier_rows:
                warnings.append(
                    f"{int(outlier_rows)} rows have at least one DEA variable more than 3 log-standard-deviations from the mean. "
                    "Check for outliers or unit-of-measure errors."
                )

    if time_col is not None:
        warnings.append(
            "A time column is selected. DEA can be run pooled across DMU-period rows or separately within each period; "
            "SFA in this app remains a pooled cross-sectional frontier, not a panel model like xtfrontier or Battese-Coelli time effects."
        )

    if sfa_data_are_logged:
        warnings.append(
            "SFA variables are being treated as already logged. DEA still requires level/positive input-output columns; "
            "use separate raw DEA columns and logged SFA columns when comparing to R/Stata log-frontier specifications."
        )

    # SFA degrees-of-freedom warning.
    if n > 0 and sfa_inputs:
        cd_params = 1 + len(sfa_inputs)
        translog_params = cd_params + len(sfa_inputs) + (len(sfa_inputs) * (len(sfa_inputs) - 1)) // 2
        p = translog_params if sfa_model == "Translog" else cd_params
        if n <= p + 3:
            warnings.append(
                f"The selected {sfa_model} SFA has {p} frontier coefficients plus variance terms for {n} rows. "
                "SFA estimates may be unstable."
            )

    return errors, warnings
