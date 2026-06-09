"""Sphinx configuration for the CFTK documentation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "CFTK"
author = "Chaorong Chen"
copyright = "2026, Chaorong Chen"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "CFTK"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/ChaorongC/CFTK",
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "logo": {
        "text": "CFTK",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ChaorongC/CFTK",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
}

html_context = {
    "github_user": "ChaorongC",
    "github_repo": "CFTK",
    "github_version": "main",
    "doc_path": "docs",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

nitpicky = False
autosummary_generate = False
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Several CFTK modules import optional scientific or bioinformatics libraries at
# module import time. Keep autodoc resilient while packaging is still evolving.
autodoc_mock_imports = [
    "adjustText",
    "Bio",
    "bx",
    "finaletoolkit",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "pyBigWig",
    "pysam",
    "scipy",
    "seaborn",
    "sklearn",
    "statsmodels",
    "xgboost",
]
