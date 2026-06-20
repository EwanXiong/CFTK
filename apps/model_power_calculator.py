"""Streamlit entry point for the CFTK model-power calculator."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
if sys.platform == "darwin":
    # Override inherited invalid locale settings before importing NumPy/joblib.
    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["LC_ALL"] = "en_US.UTF-8"
    os.environ["LC_CTYPE"] = "en_US.UTF-8"
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apps.model_power_app_page import main


if __name__ == "__main__":
    main()
