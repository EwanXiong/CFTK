"""Cached runtime for the CFTK Streamlit model-power calculator."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data"
TEMPLATE_SEED = 20260617
GRID_SEED = 20260618
BLOCK_SIZE = 20
N_JOBS = 1 if sys.platform == "darwin" else 2
N_BOOTSTRAP = 300
DETECTION_ALPHA = 0.05
NULL_SIMULATIONS_BY_PRECISION = {"Fast": 20, "Standard": 50}

from analysis.model_power import load_default_model_power_reference, prepare_template_ensemble
from analysis.model_power_operating_characteristics import run_power_sample_size_grid
from apps.model_power_app_helpers import build_power_kwargs, build_template_kwargs, available_depths_from_manifest

ProgressCallback = Callable[[int, int, str], None]


@st.cache_resource(show_spinner=False, max_entries=2)
def load_available_depths(reference_dir: str) -> tuple[int | float, ...]:
    return available_depths_from_manifest(reference_dir)


@st.cache_resource(show_spinner=False, max_entries=4)
def load_reference_data(reference_dir: str, depths: tuple[int | float, ...], sd_stat: str):
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
    cpg_std_summary, cpg_mean = load_reference_data(reference_dir, depths, sd_stat)
    return prepare_template_ensemble(
        cpg_std_summary,
        cpg_mean,
        n_templates=n_templates,
        template_kwargs=build_template_kwargs(
            depths=depths,
            n_features=n_features,
            n_signal_cpgs=n_signal_cpgs,
            meth_diff=meth_diff,
            effect_sd=effect_sd,
            effect_direction=effect_direction,
            sd_stat=sd_stat,
            within_block_rho=within_block_rho,
            block_size=block_size,
        ),
        random_state=template_seed,
    )


def _notify(callback: ProgressCallback | None, completed: int, total: int, message: str) -> None:
    if callback is not None:
        callback(completed, total, message)


def _sample_seed(grid_seed: int, sample_size: int) -> int:
    return int(np.random.SeedSequence([grid_seed, sample_size]).generate_state(1, dtype=np.uint32)[0])


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
    null_simulations_per_template: int,
    alpha: float,
    ci_method: str,
    n_bootstrap: int,
    grid_seed: int,
    _progress_callback: ProgressCallback | None = None,
):
    started = time.perf_counter()
    total_steps = len(sample_sizes) + 2
    _notify(_progress_callback, 0, total_steps, "Loading reference data and preparing CpG templates...")
    templates = prepare_templates(
        reference_dir, depths, n_features, n_signal_cpgs, meth_diff, effect_sd,
        effect_direction, sd_stat, within_block_rho, block_size, n_templates, template_seed,
    )
    _notify(_progress_callback, 1, total_steps, f"Prepared {n_templates} CpG templates.")
    power_kwargs = build_power_kwargs(
        ratio=ratio,
        cv_folds=cv_folds,
        top_k=top_k,
        min_observed_fraction=min_observed_fraction,
        target_auc=target_auc,
        specificity_target=specificity_target,
    )
    per_size_fits = n_templates * (simulations_per_template + null_simulations_per_template) * len(depths) * cv_folds
    results = []
    for index, sample_size in enumerate(sample_sizes, start=1):
        _notify(
            _progress_callback,
            index,
            total_steps,
            f"N={sample_size} ({index}/{len(sample_sizes)}): signal studies and full-pipeline null calibration (~{per_size_fits:,} CV fits).",
        )
        results.append(
            run_power_sample_size_grid(
                tuple(templates),
                (sample_size,),
                simulations_per_template=simulations_per_template,
                null_simulations_per_template=null_simulations_per_template,
                power_kwargs=power_kwargs,
                alpha=alpha,
                ci_method=ci_method,
                n_bootstrap=n_bootstrap,
                n_jobs=N_JOBS,
                random_state=_sample_seed(grid_seed, sample_size),
            )
        )
        _notify(_progress_callback, index + 1, total_steps, f"Completed N={sample_size}.")

    _notify(_progress_callback, total_steps - 1, total_steps, "Combining operating-characteristic summaries...")
    output = {
        "replicate_results": pd.concat([x["replicate_results"] for x in results], ignore_index=True),
        "null_replicate_results": pd.concat([x["null_replicate_results"] for x in results], ignore_index=True),
        "template_summary": pd.concat([x["template_summary"] for x in results], ignore_index=True),
        "power_curve": pd.concat([x["power_curve"] for x in results], ignore_index=True)
            .sort_values(["model", "mean_depth", "sample_size"]).reset_index(drop=True),
        "run_metadata": {
            "analysis_mode": "cv_discovery_null_calibrated",
            "sample_sizes": list(sample_sizes),
            "n_templates": n_templates,
            "simulations_per_template": simulations_per_template,
            "null_simulations_per_template": null_simulations_per_template,
            "alpha": alpha,
            "ci_method": ci_method,
            "n_bootstrap": n_bootstrap if ci_method == "hierarchical_bootstrap" else None,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    _notify(_progress_callback, total_steps, total_steps, "Calculation complete.")
    return output
