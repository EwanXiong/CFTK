from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis.model_power import prepare_template_ensemble
from analysis.model_power_operating_characteristics import (
    make_null_model_power_template,
    run_power_sample_size_grid,
)


def _reference_data(n_cpgs=100):
    rng = np.random.default_rng(41)
    index = [f"chr1_{i + 1}" for i in range(n_cpgs)]
    mean = pd.Series(rng.uniform(0.15, 0.85, n_cpgs), index=index)
    std = pd.DataFrame({"10_mean": rng.uniform(0.04, 0.08, n_cpgs)}, index=index)
    return std, mean


def _templates(n_templates=2):
    std, mean = _reference_data()
    return prepare_template_ensemble(
        std,
        mean,
        n_templates=n_templates,
        template_kwargs={
            "depth": [10],
            "n_features": 50,
            "n_signal_cpgs": 8,
            "meth_diff": 0.06,
            "effect_sd": 0.01,
        },
        random_state=42,
    )


def test_null_template_removes_case_control_signal():
    template = _templates(1)[0]["template"]
    null = make_null_model_power_template(template)

    assert not null.is_signal.any()
    assert np.all(null.raw_effect == 0)
    for depth in null.depths:
        np.testing.assert_allclose(null.alpha_case[depth], null.alpha_control[depth])
        np.testing.assert_allclose(null.beta_case[depth], null.beta_control[depth])
        assert np.all(null.standardized_effect[depth] == 0)


def test_null_simulation_count_must_support_requested_alpha():
    with pytest.raises(ValueError, match="must be at least 19"):
        run_power_sample_size_grid(
            _templates(1),
            [30],
            simulations_per_template=2,
            null_simulations_per_template=18,
            power_kwargs={"cv_folds": 3, "top_k": 5, "target_auc": 0.70},
            alpha=0.05,
            n_jobs=1,
        )


def test_three_power_definitions_are_bounded_and_joint_is_conservative():
    result = run_power_sample_size_grid(
        _templates(2),
        [30],
        simulations_per_template=3,
        null_simulations_per_template=19,
        power_kwargs={
            "models": ("logreg",),
            "cv_folds": 3,
            "top_k": 5,
            "target_auc": 0.70,
        },
        alpha=0.05,
        ci_method="none",
        n_jobs=1,
        random_state=43,
    )

    curve = result["power_curve"]
    row = curve.iloc[0]
    for column in (
        "detection_power",
        "target_attainment_probability",
        "probability_of_success",
    ):
        assert 0 <= row[column] <= 1

    assert row["probability_of_success"] <= row["detection_power"]
    assert row["probability_of_success"] <= row["target_attainment_probability"]
    assert row["power"] == row["probability_of_success"]
    assert row["total_null_simulations"] == 2 * 19

    replicates = result["replicate_results"]
    assert replicates["detection_pvalue"].between(0, 1).all()
    expected_joint = (
        replicates["detection_success"]
        & replicates["target_attainment_success"]
    )
    assert np.array_equal(
        replicates["probability_of_success_success"].to_numpy(),
        expected_joint.to_numpy(),
    )


def test_ci_columns_are_metric_specific():
    result = run_power_sample_size_grid(
        _templates(2),
        [30],
        simulations_per_template=2,
        null_simulations_per_template=19,
        power_kwargs={"cv_folds": 3, "top_k": 5, "target_auc": 0.70},
        alpha=0.05,
        ci_method="hierarchical_bootstrap",
        n_bootstrap=100,
        n_jobs=1,
        random_state=44,
    )
    curve = result["power_curve"]
    for metric in (
        "detection_power",
        "target_attainment_probability",
        "probability_of_success",
    ):
        assert np.isfinite(curve[f"{metric}_ci_low"]).all()
        assert np.isfinite(curve[f"{metric}_ci_high"]).all()
        assert curve[f"{metric}_ci_low"].between(0, 1).all()
        assert curve[f"{metric}_ci_high"].between(0, 1).all()
