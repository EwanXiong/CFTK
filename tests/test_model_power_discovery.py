from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.model_power import prepare_template_ensemble
from analysis.model_power_discovery import (
    _prepare_fold_features,
    run_cv_discovery_power_analysis,
    run_power_sample_size_grid,
)


def _reference_data(n_cpgs=120):
    rng = np.random.default_rng(1)
    index = [f"chr1_{index + 1}" for index in range(n_cpgs)]
    mean = pd.Series(rng.uniform(0.15, 0.85, n_cpgs), index=index)
    std = pd.DataFrame(
        {
            "10_mean": rng.uniform(0.04, 0.08, n_cpgs),
            "30_mean": rng.uniform(0.03, 0.06, n_cpgs),
        },
        index=index,
    )
    return std, mean


def test_float32_feature_ranking_does_not_overflow_when_replacing_inf():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    X_train = np.array(
        [
            [0.0, 0.1],
            [0.0, 0.2],
            [0.0, 0.3],
            [1.0, 0.4],
            [1.0, 0.5],
            [1.0, 0.6],
        ],
        dtype=np.float32,
    )
    X_test = X_train[:2].copy()
    is_signal = np.array([True, False])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _prepare_fold_features(
            X_train,
            y,
            X_test,
            is_signal,
            min_observed_fraction=0.5,
            top_k=1,
        )

    messages = [str(item.message) for item in caught]
    assert not any("overflow encountered in cast" in message for message in messages)
    assert result["X_train"].shape[1] == 1


def test_cv_discovery_power_grid_returns_bounded_power():
    std, mean = _reference_data()
    templates = prepare_template_ensemble(
        std,
        mean,
        n_templates=2,
        template_kwargs={
            "depth": [10, 30],
            "n_features": 60,
            "n_signal_cpgs": 8,
            "meth_diff": 0.05,
            "effect_sd": 0.01,
        },
        random_state=4,
    )
    result = run_power_sample_size_grid(
        templates,
        sample_sizes=[30],
        simulations_per_template=2,
        power_kwargs={
            "models": ("logreg",),
            "cv_folds": 3,
            "top_k": 5,
            "target_auc": 0.70,
        },
        ci_method="none",
        n_jobs=1,
        random_state=5,
    )
    curve = result["power_curve"]
    assert len(curve) == 2
    assert curve["power"].between(0, 1).all()
    assert curve["n_templates"].eq(2).all()


def test_thread_parallel_backend_is_explicit():
    std, mean = _reference_data()
    template = prepare_template_ensemble(
        std,
        mean,
        n_templates=1,
        template_kwargs={
            "depth": [10],
            "n_features": 40,
            "n_signal_cpgs": 6,
            "meth_diff": 0.04,
            "effect_sd": 0.0,
        },
        random_state=7,
    )[0]["template"]

    result = run_cv_discovery_power_analysis(
        template,
        total_sample_size=24,
        cv_folds=3,
        top_k=5,
        n_simulations=2,
        n_jobs=2,
        parallel_prefer="threads",
        random_state=8,
    )

    assert result["analysis_metadata"]["parallel_backend"] == "threading"


def test_negative_effect_direction_assigns_only_negative_signal_effects():
    std, mean = _reference_data()
    template = prepare_template_ensemble(
        std,
        mean,
        n_templates=1,
        template_kwargs={
            "depth": [10],
            "n_features": 50,
            "n_signal_cpgs": 6,
            "meth_diff": 0.04,
            "effect_sd": 0.0,
            "effect_direction": "negative",
        },
        random_state=12,
    )[0]["template"]

    signal_effects = template.raw_effect[template.is_signal]
    null_effects = template.raw_effect[~template.is_signal]
    assert np.all(signal_effects < 0)
    assert np.all(null_effects == 0)


def test_power_grid_preserves_equal_template_weighting():
    std, mean = _reference_data()
    templates = prepare_template_ensemble(
        std,
        mean,
        n_templates=3,
        template_kwargs={
            "depth": [10],
            "n_features": 50,
            "n_signal_cpgs": 6,
            "meth_diff": 0.04,
            "effect_sd": 0.0,
        },
        random_state=13,
    )

    result = run_power_sample_size_grid(
        templates,
        sample_sizes=[30],
        simulations_per_template=2,
        power_kwargs={
            "models": ("logreg",),
            "cv_folds": 3,
            "top_k": 5,
            "target_auc": 0.70,
        },
        ci_method="none",
        n_jobs=1,
        random_state=14,
    )

    curve = result["power_curve"]
    replicate = result["replicate_results"]
    template_summary = result["template_summary"]
    expected = template_summary["target_attainment_probability"].mean()

    assert curve["n_templates"].iloc[0] == 3
    assert curve["total_simulations"].iloc[0] == 6
    assert replicate.groupby("template_id").size().nunique() == 1
    assert curve["power"].iloc[0] == expected


def test_power_grid_ci_columns_for_none_and_hierarchical_bootstrap():
    std, mean = _reference_data()
    templates = prepare_template_ensemble(
        std,
        mean,
        n_templates=2,
        template_kwargs={
            "depth": [10],
            "n_features": 50,
            "n_signal_cpgs": 6,
            "meth_diff": 0.04,
            "effect_sd": 0.0,
        },
        random_state=15,
    )
    kwargs = {
        "models": ("logreg",),
        "cv_folds": 3,
        "top_k": 5,
        "target_auc": 0.70,
    }

    no_ci = run_power_sample_size_grid(
        templates,
        sample_sizes=[30],
        simulations_per_template=2,
        power_kwargs=kwargs,
        ci_method="none",
        n_jobs=1,
        random_state=16,
    )["power_curve"]
    with_ci = run_power_sample_size_grid(
        templates,
        sample_sizes=[30],
        simulations_per_template=2,
        power_kwargs=kwargs,
        ci_method="hierarchical_bootstrap",
        n_bootstrap=100,
        n_jobs=1,
        random_state=16,
    )["power_curve"]

    assert no_ci[["power_ci_low", "power_ci_high"]].isna().all().all()
    assert np.isfinite(with_ci[["power_ci_low", "power_ci_high"]].to_numpy()).all()
    assert with_ci["power_ci_low"].between(0, 1).all()
    assert with_ci["power_ci_high"].between(0, 1).all()
