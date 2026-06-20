"""Streamlit interface for CFTK cross-validated discovery power."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.model_power import (  # noqa: E402
    load_default_model_power_reference,
    prepare_template_ensemble,
)
from analysis.model_power_discovery import run_power_sample_size_grid  # noqa: E402
from apps.model_power_app_helpers import (  # noqa: E402
    EFFECT_DIRECTION_LABELS,
    PRECISION_MODES,
    PROBABILITY_COLUMNS,
    SD_STAT_LABELS,
    AppValidationError,
    available_depths_from_manifest,
    build_power_kwargs,
    build_template_kwargs,
    case_control_ratio,
    display_table_columns,
    parse_sample_size_range,
    parse_sample_size_text,
    precision_settings,
    result_metadata,
    top_k_choices,
    top_k_training_warning,
    validate_analysis_inputs,
    validate_biomarker_inputs,
    validate_class_counts,
    validate_depth_selection,
    workload_warning,
)
from visualization.plot_model_power import plot_power_by_sample_size  # noqa: E402

REFERENCE_DIR = ROOT / "data"
TEMPLATE_SEED = 20260617
GRID_SEED = 20260618
BLOCK_SIZE = 20
N_JOBS = 2
N_BOOTSTRAP = 300

ProgressCallback = Callable[[int, int, str], None]

st.set_page_config(
    page_title="CFTK Model-Development Power Calculator",
    layout="wide",
)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_available_depths(reference_dir: str) -> tuple[int | float, ...]:
    """Read available depth labels without loading the reference arrays."""
    return available_depths_from_manifest(reference_dir)


@st.cache_resource(show_spinner=False, max_entries=4)
def load_reference_data(
    reference_dir: str,
    depths: tuple[int | float, ...],
    sd_stat: str,
):
    """Load only the selected depth/stat reference arrays."""
    loaded = load_default_model_power_reference(
        reference_dir=reference_dir,
        depths=depths,
        sd_stats=(sd_stat,),
        include_index=False,
        mmap_mode="r",
    )
    return loaded.cpg_std_summary, loaded.cpg_mean


@st.cache_resource(show_spinner=False, max_entries=16)
def prepare_templates(
    reference_dir: str,
    depths: tuple[int | float, ...],
    n_features: int,
    n_signal_cpgs: int,
    meth_diff: float,
    effect_sd: float,
    effect_direction: str,
    sd_stat: str,
    within_block_rho: float,
    block_size: int,
    n_templates: int,
    template_seed: int,
):
    """Generate a cached template ensemble keyed by biological assumptions."""
    cpg_std_summary, cpg_mean = load_reference_data(reference_dir, depths, sd_stat)
    template_kwargs = build_template_kwargs(
        depths=depths,
        n_features=n_features,
        n_signal_cpgs=n_signal_cpgs,
        meth_diff=meth_diff,
        effect_sd=effect_sd,
        effect_direction=effect_direction,
        sd_stat=sd_stat,
        within_block_rho=within_block_rho,
        block_size=block_size,
    )
    return prepare_template_ensemble(
        cpg_std_summary,
        cpg_mean,
        n_templates=n_templates,
        template_kwargs=template_kwargs,
        random_state=template_seed,
    )


def _notify(
    callback: ProgressCallback | None,
    completed: int,
    total: int,
    message: str,
) -> None:
    if callback is not None:
        callback(completed, total, message)


@st.cache_data(show_spinner=False, max_entries=32, ttl=24 * 60 * 60)
def calculate_power_grid(
    reference_dir: str,
    depths: tuple[int | float, ...],
    n_features: int,
    n_signal_cpgs: int,
    meth_diff: float,
    effect_sd: float,
    effect_direction: str,
    sd_stat: str,
    within_block_rho: float,
    block_size: int,
    n_templates: int,
    template_seed: int,
    sample_sizes: tuple[int, ...],
    ratio: float,
    cv_folds: int,
    top_k: int,
    min_observed_fraction: float,
    target_auc: float,
    specificity_target: float,
    simulations_per_template: int,
    ci_method: str,
    n_bootstrap: int,
    grid_seed: int,
    _progress_callback: ProgressCallback | None = None,
):
    """Run the cached grid while reporting progress by sample-size point."""
    started = time.perf_counter()
    total_steps = len(sample_sizes) + 2

    _notify(
        _progress_callback,
        0,
        total_steps,
        "Loading reference data and preparing CpG templates...",
    )
    templates = prepare_templates(
        reference_dir,
        depths,
        n_features,
        n_signal_cpgs,
        meth_diff,
        effect_sd,
        effect_direction,
        sd_stat,
        within_block_rho,
        block_size,
        n_templates,
        template_seed,
    )
    _notify(
        _progress_callback,
        1,
        total_steps,
        f"Prepared {n_templates} CpG templates.",
    )

    power_kwargs = build_power_kwargs(
        ratio=ratio,
        cv_folds=cv_folds,
        top_k=top_k,
        min_observed_fraction=min_observed_fraction,
        target_auc=target_auc,
        specificity_target=specificity_target,
    )

    results = []
    per_size_fits = (
        n_templates
        * simulations_per_template
        * len(depths)
        * cv_folds
    )
    for index, sample_size in enumerate(sample_sizes, start=1):
        _notify(
            _progress_callback,
            index,
            total_steps,
            (
                f"Simulating total N={sample_size} ({index}/{len(sample_sizes)}): "
                f"approximately {per_size_fits:,} cross-validation fits."
            ),
        )
        result = run_power_sample_size_grid(
            tuple(templates),
            (sample_size,),
            simulations_per_template=simulations_per_template,
            power_kwargs=power_kwargs,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
            n_jobs=N_JOBS,
            random_state=int(
                pd.util.hash_array(
                    pd.array([grid_seed, sample_size], dtype="int64").to_numpy()
                )[0]
            ),
        )
        results.append(result)
        _notify(
            _progress_callback,
            index + 1,
            total_steps,
            f"Completed total N={sample_size}.",
        )

    _notify(
        _progress_callback,
        total_steps - 1,
        total_steps,
        "Combining simulation results and calculating power summaries...",
    )
    replicate_results = pd.concat(
        [result["replicate_results"] for result in results],
        ignore_index=True,
    )
    template_summary = pd.concat(
        [result["template_summary"] for result in results],
        ignore_index=True,
    )
    power_curve = (
        pd.concat([result["power_curve"] for result in results], ignore_index=True)
        .sort_values(["model", "mean_depth", "sample_size"])
        .reset_index(drop=True)
    )

    _notify(_progress_callback, total_steps, total_steps, "Calculation complete.")
    return {
        "replicate_results": replicate_results,
        "template_summary": template_summary,
        "power_curve": power_curve,
        "run_metadata": {
            "analysis_mode": "cv_discovery",
            "sample_sizes": list(sample_sizes),
            "n_templates": n_templates,
            "simulations_per_template": simulations_per_template,
            "total_study_simulations": (
                n_templates * len(sample_sizes) * simulations_per_template
            ),
            "ci_method": ci_method,
            "confidence": 0.95 if ci_method != "none" else None,
            "n_bootstrap": n_bootstrap if ci_method == "hierarchical_bootstrap" else None,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def _format_summary_table(curve: pd.DataFrame, *, include_ci: bool) -> pd.DataFrame:
    columns = display_table_columns(curve.columns, include_ci=include_ci)
    table = curve.loc[:, columns].copy()
    for column in PROBABILITY_COLUMNS:
        if column in table.columns:
            table[column] = table[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )
    return table


def _settings_download(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, indent=2, sort_keys=True) + "\n"


def _parse_sample_sizes_from_form(
    mode: str,
    text: str,
    start: int,
    stop: int,
    step: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if mode == "Comma-separated list":
        parsed = parse_sample_size_text(text)
    else:
        parsed = parse_sample_size_range(start=start, stop=stop, step=step)
    return parsed.values, parsed.warnings


def _render_downloads(
    *,
    curve: pd.DataFrame,
    replicate_results: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    left, middle, right = st.columns(3)
    with left:
        st.download_button(
            "Download power-curve CSV",
            curve.to_csv(index=False),
            "cftk_model_power_curve.csv",
            "text/csv",
        )
    with middle:
        with st.expander("Replicate-level output"):
            st.download_button(
                "Download replicate-level CSV",
                replicate_results.to_csv(index=False),
                "cftk_model_power_replicates.csv",
                "text/csv",
            )
    with right:
        st.download_button(
            "Download settings JSON",
            _settings_download(metadata),
            "cftk_model_power_settings.json",
            "application/json",
        )


def main() -> None:
    st.title("Model-Development Power Calculator")
    st.caption(
        "Estimates P(out-of-fold cross-validated AUC >= target AUC) for an "
        "internal biomarker-discovery cohort."
    )

    try:
        available_depths = load_available_depths(str(REFERENCE_DIR))
    except Exception as exc:  # pragma: no cover - defensive UI path
        st.error(f"Could not read model-power depth manifest: {exc}")
        st.stop()

    with st.form("model_power_settings"):
        st.subheader("1. Study Design")
        design_left, design_right = st.columns(2)
        with design_left:
            sample_input_mode = st.radio(
                "Total sample sizes",
                ["Comma-separated list", "Start / stop / step"],
                horizontal=True,
            )
            sample_text = st.text_input(
                "Sample-size list",
                value="50, 100, 150, 200",
                disabled=sample_input_mode != "Comma-separated list",
            )
            range_cols = st.columns(3)
            with range_cols[0]:
                sample_start = st.number_input(
                    "Start", min_value=1, value=50, step=10,
                    disabled=sample_input_mode != "Start / stop / step",
                )
            with range_cols[1]:
                sample_stop = st.number_input(
                    "Stop", min_value=1, value=200, step=10,
                    disabled=sample_input_mode != "Start / stop / step",
                )
            with range_cols[2]:
                sample_step = st.number_input(
                    "Step", min_value=1, value=50, step=10,
                    disabled=sample_input_mode != "Start / stop / step",
                )
        with design_right:
            ratio_label = st.selectbox(
                "Case-to-control ratio",
                ["1:3", "1:2", "1:1", "2:1", "3:1", "Custom"],
                index=2,
            )
            custom_ratio = st.number_input(
                "Custom n_cases / n_controls",
                min_value=0.01,
                value=1.0,
                step=0.05,
                format="%.2f",
                disabled=ratio_label != "Custom",
            )
            default_depths = tuple(
                depth for depth in (10, 30) if depth in set(available_depths)
            )
            selected_depths = st.multiselect(
                "Mean sequencing depths",
                available_depths,
                default=default_depths or available_depths[:1],
                format_func=lambda value: f"{value:g}x",
            )
            target_auc = st.slider(
                "Target cross-validated AUC",
                min_value=0.60,
                max_value=0.90,
                value=0.75,
                step=0.01,
            )

        st.subheader("2. Biomarker Assumptions")
        biomarker_left, biomarker_right = st.columns(2)
        with biomarker_left:
            n_features = st.selectbox(
                "Number of candidate CpGs", [100, 250, 500, 1000, 2000], index=2
            )
            n_signal_cpgs = st.number_input(
                "Number of true signal CpGs",
                min_value=1,
                max_value=int(n_features),
                value=min(20, int(n_features)),
                step=1,
            )
            meth_diff = st.slider(
                "Mean absolute methylation difference (absolute beta-value difference)",
                min_value=0.01,
                max_value=0.15,
                value=0.05,
                step=0.005,
                format="%.3f",
            )
        with biomarker_right:
            available_top_k = top_k_choices(int(n_features))
            top_k = st.selectbox(
                "Selected CpGs per training fold",
                available_top_k,
                index=available_top_k.index(10) if 10 in available_top_k else 0,
            )

        with st.expander("3. Advanced Settings", expanded=False):
            adv_left, adv_right = st.columns(2)
            with adv_left:
                effect_sd = st.slider(
                    "Effect-size SD", 0.0, 0.05, 0.015, 0.005, format="%.3f"
                )
                effect_label = st.selectbox(
                    "Effect direction", list(EFFECT_DIRECTION_LABELS), index=0
                )
                within_block_rho = st.slider(
                    "Within-block correlation", 0.0, 0.8, 0.20, 0.05
                )
                cv_folds = st.selectbox(
                    "Cross-validation folds", [3, 5, 10], index=1
                )
            with adv_right:
                min_observed_fraction = st.slider(
                    "Minimum observed fraction", 0.30, 0.90, 0.50, 0.05
                )
                specificity_target = st.slider(
                    "Specificity operating point", 0.80, 0.99, 0.90, 0.01
                )
                sd_label = st.selectbox(
                    "SD uncertainty scenario",
                    list(SD_STAT_LABELS),
                    index=0,
                    help=(
                        "Lower SD is generally optimistic; upper SD is generally "
                        "conservative."
                    ),
                )

        st.subheader("4. Computation and Display Settings")
        comp_left, comp_middle, comp_right = st.columns(3)
        with comp_left:
            precision_mode = st.selectbox(
                "Precision mode", list(PRECISION_MODES), index=0
            )
        with comp_middle:
            calculate_ci = st.checkbox(
                "Calculate confidence interval", value=False
            )
        with comp_right:
            plot_width = st.slider(
                "Plot width (inches)",
                min_value=6.0,
                max_value=14.0,
                value=9.0,
                step=0.5,
                help="Controls the rendered and exported Matplotlib figure width.",
            )

        submitted = st.form_submit_button("Calculate", type="primary")

    if not submitted:
        return

    try:
        precision = precision_settings(precision_mode)
        sample_sizes, sample_warnings = _parse_sample_sizes_from_form(
            sample_input_mode,
            sample_text,
            int(sample_start),
            int(sample_stop),
            int(sample_step),
        )
        ratio = case_control_ratio(
            ratio_label,
            custom_ratio=float(custom_ratio) if ratio_label == "Custom" else None,
        )
        depths = validate_depth_selection(selected_depths)
        effect_direction = EFFECT_DIRECTION_LABELS[effect_label]
        sd_stat = SD_STAT_LABELS[sd_label]
        ci_method = "hierarchical_bootstrap" if calculate_ci else "none"
        n_bootstrap = N_BOOTSTRAP if calculate_ci else 0

        validate_class_counts(
            sample_sizes=sample_sizes, ratio=ratio, cv_folds=int(cv_folds)
        )
        validate_biomarker_inputs(
            n_features=int(n_features),
            n_signal_cpgs=int(n_signal_cpgs),
            top_k=int(top_k),
            meth_diff=float(meth_diff),
            effect_sd=float(effect_sd),
            within_block_rho=float(within_block_rho),
            sd_stat=sd_stat,
            effect_direction=effect_direction,
        )
        validate_analysis_inputs(
            target_auc=float(target_auc),
            min_observed_fraction=float(min_observed_fraction),
            specificity_target=float(specificity_target),
            cv_folds=int(cv_folds),
        )

        for message in sample_warnings:
            st.warning(message)
        top_k_warning = top_k_training_warning(
            top_k=int(top_k),
            sample_sizes=sample_sizes,
            ratio=ratio,
            cv_folds=int(cv_folds),
        )
        if top_k_warning:
            st.warning(top_k_warning)
        large_warning = workload_warning(
            sample_sizes=sample_sizes,
            depths=depths,
            n_templates=precision["n_templates"],
            simulations_per_template=precision["simulations_per_template"],
        )
        if large_warning:
            st.warning(large_warning)

        total_grid_jobs = len(sample_sizes) * precision["n_templates"]
        st.caption(
            f"Planned workload: {len(sample_sizes)} sample-size points × "
            f"{precision['n_templates']} templates = {total_grid_jobs} grid jobs; "
            f"{precision['simulations_per_template']} Monte Carlo studies per template."
        )
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(completed: int, total: int, message: str) -> None:
            fraction = 0.0 if total <= 0 else min(max(completed / total, 0.0), 1.0)
            progress_bar.progress(fraction)
            progress_text.caption(
                f"Step {min(completed, total)}/{total}: {message}"
            )

        result = calculate_power_grid(
            str(REFERENCE_DIR),
            tuple(depths),
            int(n_features),
            int(n_signal_cpgs),
            float(meth_diff),
            float(effect_sd),
            effect_direction,
            sd_stat,
            float(within_block_rho),
            BLOCK_SIZE,
            precision["n_templates"],
            TEMPLATE_SEED,
            tuple(sample_sizes),
            float(ratio),
            int(cv_folds),
            int(top_k),
            float(min_observed_fraction),
            float(target_auc),
            float(specificity_target),
            precision["simulations_per_template"],
            ci_method,
            n_bootstrap,
            GRID_SEED,
            _progress_callback=update_progress,
        )
        progress_bar.progress(1.0)
        progress_text.success("Calculation complete.")

        curve = result["power_curve"]
        st.subheader("Results")
        fig, ax = plt.subplots(figsize=(float(plot_width), 5.5))
        fig, ax = plot_power_by_sample_size(
            curve,
            show_ci=calculate_ci,
            target_power=0.80,
            title="Model-development power by study size and sequencing depth",
            ax=ax,
        )
        ax.set_ylabel("Probability of reaching target CV AUC")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        has_ci = (
            calculate_ci
            and {"power_ci_low", "power_ci_high"}.issubset(curve.columns)
            and not curve[["power_ci_low", "power_ci_high"]].isna().all().all()
        )
        st.dataframe(
            _format_summary_table(curve, include_ci=has_ci),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            "- `power` is the proportion of simulated studies whose out-of-fold "
            "CV AUC reaches the selected target.\n"
            "- Feature recall and precision are simulation diagnostics because "
            "the true signal CpGs are known in the simulation.\n"
            "- Selection Jaccard summarizes fold-to-fold feature-set overlap.\n"
            "- This result evaluates internal model development only and does "
            "not estimate external cohort performance."
        )

        user_inputs = {
            "sample_sizes": list(sample_sizes),
            "case_to_control_ratio": ratio_label,
            "ratio": float(ratio),
            "depths": list(depths),
            "target_auc": float(target_auc),
            "n_features": int(n_features),
            "n_signal_cpgs": int(n_signal_cpgs),
            "meth_diff": float(meth_diff),
            "top_k": int(top_k),
            "effect_sd": float(effect_sd),
            "effect_direction": effect_direction,
            "within_block_rho": float(within_block_rho),
            "cv_folds": int(cv_folds),
            "min_observed_fraction": float(min_observed_fraction),
            "specificity_target": float(specificity_target),
            "sd_stat": sd_stat,
            "calculate_ci": bool(calculate_ci),
            "plot_width_inches": float(plot_width),
        }
        metadata = result_metadata(
            user_inputs=user_inputs,
            precision_mode=precision_mode,
            template_seed=TEMPLATE_SEED,
            grid_seed=GRID_SEED,
            ci_method=ci_method,
            n_bootstrap=n_bootstrap,
        )
        metadata["run_metadata"] = result.get("run_metadata", {})
        metadata["reference"] = {
            "reference_dir": str(REFERENCE_DIR),
            "selected_depths": list(depths),
            "sd_stat": sd_stat,
        }

        _render_downloads(
            curve=curve,
            replicate_results=result["replicate_results"],
            metadata=metadata,
        )

    except AppValidationError as exc:
        st.error(str(exc))
    except Exception:  # pragma: no cover - defensive UI path
        st.error(
            "The calculation could not be completed. Check the input settings "
            "or reduce the requested workload."
        )
        print(traceback.format_exc(), file=sys.stderr)


if __name__ == "__main__":
    main()
