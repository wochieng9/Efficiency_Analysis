from __future__ import annotations

import io
import json
from typing import Optional

import pandas as pd
import streamlit as st

from efficiency_tool.ui.widgets import display_download_button

def safe_file_stem(value: str) -> str:
    """Create a simple, portable file stem for chart exports."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "chart"

def altair_chart_to_json_bytes(chart: object) -> bytes:
    """Serialize an Altair chart as a Vega-Lite JSON specification."""
    try:
        spec_json = chart.to_json(indent=2)
    except TypeError:
        spec_json = chart.to_json()
    except Exception:
        spec_json = json.dumps(chart.to_dict(), indent=2)
    return spec_json.encode("utf-8")

def altair_chart_to_html_bytes(chart: object) -> bytes:
    """Serialize an Altair chart as a standalone interactive HTML document."""
    try:
        html = chart.to_html()
    except Exception:
        buffer = io.StringIO()
        chart.save(buffer, format="html")
        html = buffer.getvalue()
    return str(html).encode("utf-8")

def altair_png_from_json(spec_json: str) -> Optional[bytes]:
    """Convert Vega-Lite JSON to PNG when vl-convert-python is available."""
    try:
        import vl_convert as vlc  # type: ignore

        return vlc.vegalite_to_png(spec_json, scale=2.0)
    except Exception:
        return None

def altair_svg_from_json(spec_json: str) -> Optional[bytes]:
    """Convert Vega-Lite JSON to SVG when vl-convert-python is available."""
    try:
        import vl_convert as vlc  # type: ignore

        svg = vlc.vegalite_to_svg(spec_json)
        return str(svg).encode("utf-8")
    except Exception:
        return None

def render_altair_chart_with_downloads(
    chart: object,
    file_stem: str,
    data: Optional[pd.DataFrame] = None,
) -> None:
    """Render an interactive Altair chart and offer reusable figure download buttons."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

    safe_stem = safe_file_stem(file_stem)
    try:
        json_bytes = altair_chart_to_json_bytes(chart)
        html_bytes = altair_chart_to_html_bytes(chart)
    except Exception as exc:
        st.caption(f"Figure export is unavailable for this chart: {exc}")
        return

    spec_json = json_bytes.decode("utf-8")
    png_bytes = altair_png_from_json(spec_json)
    svg_bytes = altair_svg_from_json(spec_json)

    with st.expander("Download this figure", expanded=False):
        cols = st.columns(5)
        with cols[0]:
            display_download_button(
                "HTML",
                html_bytes,
                f"{safe_stem}.html",
                "text/html",
                key=f"{safe_stem}_html",
            )
        with cols[1]:
            display_download_button(
                "Vega-Lite JSON",
                json_bytes,
                f"{safe_stem}.json",
                "application/json",
                key=f"{safe_stem}_json",
            )
        with cols[2]:
            if png_bytes is not None:
                display_download_button(
                    "PNG",
                    png_bytes,
                    f"{safe_stem}.png",
                    "image/png",
                    key=f"{safe_stem}_png",
                )
            else:
                st.caption("PNG export needs vl-convert-python.")
        with cols[3]:
            if svg_bytes is not None:
                display_download_button(
                    "SVG",
                    svg_bytes,
                    f"{safe_stem}.svg",
                    "image/svg+xml",
                    key=f"{safe_stem}_svg",
                )
            else:
                st.caption("SVG export needs vl-convert-python.")
        with cols[4]:
            if data is not None and not data.empty:
                display_download_button(
                    "Chart data CSV",
                    data.to_csv(index=False).encode("utf-8"),
                    f"{safe_stem}_data.csv",
                    "text/csv",
                    key=f"{safe_stem}_data",
                )
            else:
                st.caption("No chart data export.")

