from __future__ import annotations

from typing import List, Optional, Sequence

import streamlit as st

def add_sidebar_selectbox(label: str, columns: Sequence[str], help_text: str = "") -> Optional[str]:
    """Select one optional column from a dataframe."""
    options: List[Optional[str]] = [None] + list(columns)
    return st.sidebar.selectbox(
        label,
        options=options,
        format_func=lambda x: "None" if x is None else str(x),
        help=help_text,
    )

def display_download_button(
    label: str,
    data: bytes,
    file_name: str,
    mime: str,
    key: Optional[str] = None,
) -> None:
    """Use on_click='ignore' when supported, with a fallback for older Streamlit versions."""
    kwargs = {
        "label": label,
        "data": data,
        "file_name": file_name,
        "mime": mime,
    }
    if key is not None:
        kwargs["key"] = key

    try:
        st.download_button(**kwargs, on_click="ignore")
    except TypeError:
        st.download_button(**kwargs)

