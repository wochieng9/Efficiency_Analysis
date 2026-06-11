from __future__ import annotations

import io
from typing import Dict

import pandas as pd

def make_excel_download(tables: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)
    return buffer.getvalue()

