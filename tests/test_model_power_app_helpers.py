import ast
import json
from pathlib import Path

import pytest


def test_parse_sample_size_text_sorts_and_deduplicates():
    from apps.model_power_app_helpers import parse_sample_size_text

    parsed = parse_sample_size_text("100, 50, 100, 200")
    assert parsed.values == (50, 100, 200)
    assert parsed.warnings == ("Duplicate sample sizes were removed.",)


@pytest.mark.parametrize("text", ["", "10, abc", "0, 20", "1,2,3,4,5,6,7,8,9"])
def test_parse_sample_size_text_rejects_invalid_inputs(text):
    from apps.model_power_app_helpers import AppValidationError, parse_sample_size_text

    with pytest.raises(AppValidationError):
        parse_sample_size_text(text)


def test_parse_sample_size_text_rejects_public_sample_size_limit():
    from apps.model_power_app_helpers import AppValidationError, parse_sample_size_text

    with pytest.raises(AppValidationError, match="supports total sample sizes up to"):
        parse_sample_size_text("2001")


def test_parse_sample_size_range_generates_sorted_grid():
    from apps.model_power_app_helpers import parse_sample_size_range

    parsed = parse_sample_size_range(start=50, stop=100, step=25)
    assert parsed.values == (50, 75, 100)


def test_case_control_ratio_conversion():
    from apps.model_power_app_helpers import case_control_ratio

    assert case_control_ratio("1:3") == pytest.approx(1 / 3)
    assert case_control_ratio("1:1") == pytest.approx(1.0)
    assert case_control_ratio("3:1") == pytest.approx(3.0)
    assert case_control_ratio("Custom", custom_ratio=1.25) == pytest.approx(1.25)


def test_class_count_validation_requires_each_class_per_fold():
    from apps.model_power_app_helpers import AppValidationError, validate_class_counts

    validate_class_counts(sample_sizes=(30,), ratio=1.0, cv_folds=5)
    with pytest.raises(AppValidationError, match="at least 10 cases and 10 controls"):
        validate_class_counts(sample_sizes=(12,), ratio=1.0, cv_folds=10)


def test_available_depths_are_read_from_manifest(tmp_path):
    from apps.model_power_app_helpers import available_depths_from_manifest

    manifest = {"depth_labels": ["5", "10", "30"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert available_depths_from_manifest(tmp_path) == (5, 10, 30)


def test_biomarker_validation_bounds_signal_and_top_k():
    from apps.model_power_app_helpers import AppValidationError, validate_biomarker_inputs

    validate_biomarker_inputs(
        n_features=100,
        n_signal_cpgs=20,
        top_k=10,
        meth_diff=0.05,
        effect_sd=0.015,
        within_block_rho=0.2,
        sd_stat="mean",
    )
    with pytest.raises(AppValidationError, match="n_signal_cpgs"):
        validate_biomarker_inputs(
            n_features=100,
            n_signal_cpgs=101,
            top_k=10,
            meth_diff=0.05,
            effect_sd=0.015,
            within_block_rho=0.2,
            sd_stat="mean",
        )
    with pytest.raises(AppValidationError, match="top_k"):
        validate_biomarker_inputs(
            n_features=100,
            n_signal_cpgs=20,
            top_k=101,
            meth_diff=0.05,
            effect_sd=0.015,
            within_block_rho=0.2,
            sd_stat="mean",
        )


def test_biomarker_validation_rejects_unsupported_sd_and_effect_direction():
    from apps.model_power_app_helpers import AppValidationError, validate_biomarker_inputs

    with pytest.raises(AppValidationError, match="Unsupported SD"):
        validate_biomarker_inputs(
            n_features=100,
            n_signal_cpgs=20,
            top_k=10,
            meth_diff=0.05,
            effect_sd=0.015,
            within_block_rho=0.2,
            sd_stat="bad",
        )
    with pytest.raises(AppValidationError, match="Unsupported effect direction"):
        validate_biomarker_inputs(
            n_features=100,
            n_signal_cpgs=20,
            top_k=10,
            meth_diff=0.05,
            effect_sd=0.015,
            within_block_rho=0.2,
            sd_stat="mean",
            effect_direction="mixed",
        )


def test_analysis_validation_bounds_secondary_settings():
    from apps.model_power_app_helpers import AppValidationError, validate_analysis_inputs

    validate_analysis_inputs(
        target_auc=0.75,
        min_observed_fraction=0.5,
        specificity_target=0.9,
        cv_folds=5,
    )
    with pytest.raises(AppValidationError, match="Target cross-validated AUC"):
        validate_analysis_inputs(
            target_auc=0.95,
            min_observed_fraction=0.5,
            specificity_target=0.9,
            cv_folds=5,
        )
    with pytest.raises(AppValidationError, match="Cross-validation folds"):
        validate_analysis_inputs(
            target_auc=0.75,
            min_observed_fraction=0.5,
            specificity_target=0.9,
            cv_folds=4,
        )


def test_precision_mode_mapping():
    from apps.model_power_app_helpers import precision_settings

    assert precision_settings("Fast") == {
        "n_templates": 5,
        "simulations_per_template": 10,
    }
    assert precision_settings("Standard") == {
        "n_templates": 10,
        "simulations_per_template": 20,
    }


def test_workload_limit_rejects_oversized_public_request():
    from apps.model_power_app_helpers import AppValidationError, workload_warning

    with pytest.raises(AppValidationError, match="computational limit"):
        workload_warning(
            sample_sizes=(500, 1000, 1500, 2000),
            depths=(10, 20, 30, 40),
            n_templates=10,
            simulations_per_template=20,
        )


def test_standard_default_only_warns():
    from apps.model_power_app_helpers import workload_warning

    warning = workload_warning(
        sample_sizes=(50, 100, 150, 200),
        depths=(10, 30),
        n_templates=10,
        simulations_per_template=20,
    )
    assert warning is not None


def test_app_uses_null_calibrated_operating_characteristics():
    runtime_source = Path("apps/model_power_app_runtime.py").read_text()
    runtime_tree = ast.parse(runtime_source)
    imports = [
        node
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "analysis.model_power_operating_characteristics"
        and any(alias.name == "run_power_sample_size_grid" for alias in node.names)
    ]
    assert imports

    page_source = Path("apps/model_power_app_page.py").read_text()
    assert "Probability of success (conservative)" in page_source
    assert "Target-attainment probability" in page_source
    assert "Detection power" in page_source
    assert "train_fraction" not in runtime_source + page_source


def test_no_explicit_penalty_argument_in_app_path():
    for path in [
        Path("apps/model_power_calculator.py"),
        Path("apps/model_power_app_helpers.py"),
        Path("apps/model_power_app_runtime.py"),
        Path("apps/model_power_app_page.py"),
    ]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword):
                assert node.arg != "penalty"
