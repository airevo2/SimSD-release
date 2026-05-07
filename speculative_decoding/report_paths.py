"""
 / bench

 UTC


  export SPEC_DECODE_REPORT_DATE=2025-03-18
"""

from __future__ import annotations

import os
from datetime import date


def report_date_str() -> str:
    """YYYY-MM-DD SPEC_DECODE_REPORT_DATE"""
    v = os.environ.get("SPEC_DECODE_REPORT_DATE", "").strip()
    if v:
        return v
    return date.today().isoformat()


def run_subdir(output_dir: str, prefix: str, tag: str) -> str:
    """results/{date}/{prefix}{tag}/"""
    return os.path.join(output_dir, report_date_str(), f"{prefix}{tag}")
