"""Streamlit interface for CFTK cross-validated discovery power."""

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.model_power import (
    get_default_model_power_reference,
    prepare_template_ensemble,
)
from analysis.model_power_discovery import run_power_sample_size_grid
from visualization.plot_model_power import plot_power_by_sample_size

st.set_page_config(page_title="CFTK Model Power", layout="wide")


@st.cache_resource
def load_templates():
    cpg_std_summary, cpg_mean = get_default_model_power_reference(
        depths=[10, 30], sd_stats=("mean",)
    )
    return prepare_template_ensemble(
        cpg_std_summary,
        cpg_mean,
        n_templates=5,
        template_kwargs={
            "depth": [10, 30],
            "n_features": 500,
            "n_signal_cpgs": 20,
            "meth_diff": 0.05,
            "effect_sd": 0.015,
            "effect_direction": "balanced",
            "sd_stat": "mean",
            "within_block_rho": 0.20,
            "block_size": 20,
        },
        random_state=20260617,
    )


@st.cache_data(show_spinner=False)
def calculate(sample_sizes, simulations, cv_folds, top_k, target_auc, use_ci):
    return run_power_sample_size_grid(
        load_templates(),
        sample_sizes,
        simulations_per_template=simulations,
        power_kwargs={
            "models": ("logreg",),
            "cv_folds": cv_folds,
            "cv_repeats": 1,
            "top_k": top_k,
            "target_auc": target_auc,
            "specificity_target": 0.90,
            "paired_depths": True,
        },
        ci_method="hierarchical_bootstrap" if use_ci else "none",
        n_bootstrap=300,
        n_jobs=2,
        random_state=20260618,
    )


st.title("CFTK Model-Development Power Calculator")
st.write(
    "This calculator estimates the probability that a fixed biomarker "
    "pipeline reaches a target out-of-fold cross-validated AUC. It does not "
    "estimate external generalizability."
)

with st.form("settings"):
    sample_text = st.text_input("Total sample sizes", "50, 100, 150, 200")
    cv_folds = st.selectbox("CV folds", [3, 5, 10], index=1)
    top_k = st.select_slider("Selected CpGs", [5, 10, 20, 50], value=10)
    target_auc = st.slider("Target CV AUC", 0.60, 0.90, 0.75, 0.01)
    simulations = st.select_slider(
        "Simulations per template", [5, 10, 20, 50], value=10
    )
    use_ci = st.checkbox("Calculate hierarchical confidence interval")
    submitted = st.form_submit_button("Calculate")

if submitted:
    try:
        sample_sizes = tuple(
            sorted({int(value.strip()) for value in sample_text.split(",")})
        )
        if not sample_sizes or len(sample_sizes) > 8:
            raise ValueError("Provide between one and eight sample sizes.")
        with st.spinner("Running simulations..."):
            result = calculate(
                sample_sizes, simulations, cv_folds, top_k, target_auc, use_ci
            )
        curve = result["power_curve"]
        fig, _ = plot_power_by_sample_size(curve, show_ci=use_ci)
        st.pyplot(fig, use_container_width=True)
        st.dataframe(curve, hide_index=True, use_container_width=True)
        st.download_button(
            "Download CSV",
            curve.to_csv(index=False),
            "cftk_model_power.csv",
            "text/csv",
        )
    except Exception as exc:
        st.error(str(exc))
